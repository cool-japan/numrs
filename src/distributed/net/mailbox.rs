//! In-memory per-key FIFO delivery, keyed by `(src, ctx, tag)`.
//!
//! A [`super::link::Link`]'s background reader task decodes frames off the
//! wire and pushes each decoded payload into the mailbox under the key from
//! its [`super::frame::FrameHeader`] (`src`, `ctx`, `tag`). A caller doing
//! `recv(src, ctx, tag)` then just pops that key — it is never handed a
//! different rank's or a different tag's message, which is the bug in
//! `ConnectionManager::recv` today (see the [`super`] module docs).
//!
//! # Concurrency model
//!
//! One `std::sync::Mutex` guards the whole table; it is **never** held across
//! an `.await`. A `pop` that finds its queue empty parks a
//! [`tokio::sync::oneshot`] sender in that key's waiter list and awaits the
//! receiver outside the lock. A `push` hands the item directly to the oldest
//! live waiter, falling back to the queue when there is none.
//!
//! ## The timeout/delivery race
//!
//! A naive implementation loses messages on this interleaving: a waiter's
//! `timeout` fires, and a concurrent `push` hands the payload to that
//! now-abandoned oneshot, which drops it. Both halves are closed here:
//!
//! - [`Mailbox::push`] treats `oneshot::Sender::send` returning `Err(value)`
//!   as "that waiter is gone" and moves on to the next one (or the queue) —
//!   the value is never dropped.
//! - [`Mailbox::pop_timeout`] on expiry re-takes the lock, removes *its own*
//!   waiter (identified by a unique id, not by position), and only then
//!   drains the oneshot and re-checks the queue. Because `push` performs its
//!   hand-off while holding the same lock, any payload already committed to
//!   this waiter is still recoverable at that point.
//!
//! ## Bounded growth
//!
//! Each key's queue is capped ([`Mailbox::with_key_capacity`]). Overflow is
//! an **error** ([`NetError::MailboxFull`]), never a blocking push: a reader
//! task that awaited mailbox capacity would let one un-drained key stall
//! every other key sharing that link — the exact deadlock shape this
//! transport exists to remove.

use super::NetError;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::oneshot;

/// Mailbox key: `(src rank, context id, tag)`.
///
/// Messages sharing a key are delivered in the order they were pushed
/// (FIFO). Messages under different keys have no ordering guarantee
/// relative to each other.
///
/// - `src`: the sending rank.
/// - `ctx`: the logical communicator/context id — distinguishes concurrent
///   sub-communicators (e.g. one produced by
///   [`super::super::process::Communicator::split`]) that happen to share
///   one physical [`super::link::Link`].
/// - `tag`: the user-facing message tag (mirrors
///   [`super::super::comm::MessageTag`], widened to `u64`).
pub type MailboxKey = (u32, u64, u64);

/// Default ceiling on undelivered messages queued under a single
/// [`MailboxKey`], used by [`Mailbox::new`].
///
/// [`super::endpoint::Endpoint`] scales this off
/// [`super::EndpointConfig::queue_depth`] instead of taking it verbatim; see
/// [`super::endpoint::Endpoint::key_capacity_for`].
pub const DEFAULT_KEY_CAPACITY: usize = 4096;

/// One key's state: queued payloads plus the waiters parked on that key.
struct Entry<T> {
    queue: VecDeque<T>,
    waiters: VecDeque<(u64, oneshot::Sender<T>)>,
}

impl<T> Entry<T> {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            waiters: VecDeque::new(),
        }
    }

    /// True when this entry holds nothing at all and can be reclaimed.
    fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.waiters.is_empty()
    }
}

struct Inner<T> {
    entries: HashMap<MailboxKey, Entry<T>>,
    closed: bool,
}

/// A FIFO mailbox keyed by [`MailboxKey`]. See the module docs.
pub struct Mailbox<T> {
    inner: Mutex<Inner<T>>,
    key_capacity: usize,
    next_waiter_id: AtomicU64,
}

impl<T> std::fmt::Debug for Mailbox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (keys, closed) = match self.inner.lock() {
            Ok(guard) => (guard.entries.len(), guard.closed),
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                (guard.entries.len(), guard.closed)
            }
        };
        f.debug_struct("Mailbox")
            .field("keys", &keys)
            .field("closed", &closed)
            .field("key_capacity", &self.key_capacity)
            .finish()
    }
}

impl<T> Mailbox<T> {
    /// Create an empty mailbox with [`DEFAULT_KEY_CAPACITY`] per key.
    pub fn new() -> Self {
        Self::with_key_capacity(DEFAULT_KEY_CAPACITY)
    }

    /// Create an empty mailbox capping each key's queue at `key_capacity`
    /// undelivered messages. A capacity of `0` is raised to `1` so a mailbox
    /// is never unusable.
    pub fn with_key_capacity(key_capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                closed: false,
            }),
            key_capacity: key_capacity.max(1),
            next_waiter_id: AtomicU64::new(0),
        }
    }

    /// Per-key ceiling on undelivered messages.
    pub fn key_capacity(&self) -> usize {
        self.key_capacity
    }

    /// Lock the table, recovering from a poisoned mutex rather than
    /// panicking (COOLJAPAN no-unwrap policy).
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner<T>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Push `item` onto the FIFO queue for `key`, waking any waiter blocked
    /// in [`Self::pop`] for that same key.
    ///
    /// Hands the item straight to the oldest still-live waiter when there is
    /// one; a waiter whose receiver has already gone away (cancelled or
    /// timed out) is skipped, never consumed. Returns
    /// [`NetError::MailboxFull`] when the item would have to be queued and
    /// the key is already at [`Self::key_capacity`], and
    /// [`NetError::MailboxClosed`] after [`Self::close`].
    pub fn push(&self, key: MailboxKey, item: T) -> Result<(), NetError> {
        let mut guard = self.lock();
        if guard.closed {
            return Err(NetError::MailboxClosed);
        }
        let capacity = self.key_capacity;
        let entry = guard.entries.entry(key).or_insert_with(Entry::new);

        // Hand off to the oldest live waiter, if any. `send` returning the
        // value back means that receiver is gone; try the next one rather
        // than dropping the payload.
        let mut item = item;
        while let Some((_id, waiter)) = entry.waiters.pop_front() {
            match waiter.send(item) {
                Ok(()) => return Ok(()),
                Err(returned) => item = returned,
            }
        }

        if entry.queue.len() >= capacity {
            let (src, ctx, tag) = key;
            return Err(NetError::MailboxFull {
                src,
                ctx,
                tag,
                capacity,
            });
        }
        entry.queue.push_back(item);
        Ok(())
    }

    /// Wait indefinitely for the next item queued under `key`, in FIFO order.
    ///
    /// Prefer [`Self::pop_timeout`]: an unbounded wait on a peer that never
    /// sends is indistinguishable from a hang.
    pub async fn pop(&self, key: MailboxKey) -> Result<T, NetError> {
        let rx = match self.park(key)? {
            Parked::Ready(item) => return Ok(item),
            Parked::Waiting { rx, .. } => rx,
        };
        rx.await.map_err(|_| NetError::MailboxClosed)
    }

    /// Wait up to `timeout` for the next item queued under `key`.
    ///
    /// On expiry returns [`NetError::RecvTimeout`] naming the exact
    /// `(src, ctx, tag)` that never arrived.
    pub async fn pop_timeout(&self, key: MailboxKey, timeout: Duration) -> Result<T, NetError> {
        let (mut rx, waiter_id) = match self.park(key)? {
            Parked::Ready(item) => return Ok(item),
            Parked::Waiting { rx, waiter_id } => (rx, waiter_id),
        };

        match tokio::time::timeout(timeout, &mut rx).await {
            Ok(Ok(item)) => Ok(item),
            Ok(Err(_)) => Err(NetError::MailboxClosed),
            Err(_elapsed) => {
                // Expiry and delivery can interleave. Under the lock:
                // (1) unregister this waiter so no *new* hand-off targets it,
                // (2) drain the oneshot in case a push already committed a
                //     payload to it, (3) re-check the queue.
                let mut guard = self.lock();
                let closed = guard.closed;
                let recovered = match guard.entries.get_mut(&key) {
                    Some(entry) => {
                        entry.waiters.retain(|(id, _)| *id != waiter_id);
                        let recovered = rx.try_recv().ok().or_else(|| entry.queue.pop_front());
                        if entry.is_idle() {
                            guard.entries.remove(&key);
                        }
                        recovered
                    }
                    None => rx.try_recv().ok(),
                };
                drop(guard);

                match recovered {
                    Some(item) => Ok(item),
                    None if closed => Err(NetError::MailboxClosed),
                    None => {
                        let (src, ctx, tag) = key;
                        Err(NetError::RecvTimeout {
                            src,
                            ctx,
                            tag,
                            timeout,
                        })
                    }
                }
            }
        }
    }

    /// Take an already-queued item or register a waiter, atomically.
    fn park(&self, key: MailboxKey) -> Result<Parked<T>, NetError> {
        let mut guard = self.lock();
        if guard.closed {
            return Err(NetError::MailboxClosed);
        }
        let entry = guard.entries.entry(key).or_insert_with(Entry::new);
        if let Some(item) = entry.queue.pop_front() {
            if entry.is_idle() {
                guard.entries.remove(&key);
            }
            return Ok(Parked::Ready(item));
        }
        let waiter_id = self.next_waiter_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        entry.waiters.push_back((waiter_id, tx));
        Ok(Parked::Waiting { rx, waiter_id })
    }

    /// Remove the next item queued under `key` if one is already available,
    /// without waiting.
    pub fn try_pop(&self, key: MailboxKey) -> Result<Option<T>, NetError> {
        let mut guard = self.lock();
        if guard.closed {
            return Err(NetError::MailboxClosed);
        }
        let item = match guard.entries.get_mut(&key) {
            Some(entry) => {
                let item = entry.queue.pop_front();
                if entry.is_idle() {
                    guard.entries.remove(&key);
                }
                item
            }
            None => None,
        };
        Ok(item)
    }

    /// Number of items currently queued under `key`.
    pub fn len(&self, key: MailboxKey) -> Result<usize, NetError> {
        let guard = self.lock();
        if guard.closed {
            return Err(NetError::MailboxClosed);
        }
        Ok(guard.entries.get(&key).map_or(0, |e| e.queue.len()))
    }

    /// Whether `key` currently has no queued items.
    pub fn is_empty(&self, key: MailboxKey) -> Result<bool, NetError> {
        Ok(self.len(key)? == 0)
    }

    /// Number of keys with queued items or parked waiters.
    pub fn live_keys(&self) -> usize {
        self.lock().entries.len()
    }

    /// Shut the mailbox down: every parked waiter wakes with
    /// [`NetError::MailboxClosed`] and all further operations fail the same
    /// way. Idempotent.
    pub fn close(&self) {
        let mut guard = self.lock();
        guard.closed = true;
        // Dropping the senders wakes every parked `pop` with a receive error,
        // which both `pop` and `pop_timeout` map to `MailboxClosed`.
        guard.entries.clear();
    }

    /// Whether [`Self::close`] has been called.
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }
}

/// Outcome of [`Mailbox::park`]: either an item was already queued, or this
/// caller is now registered as a waiter.
enum Parked<T> {
    Ready(T),
    Waiting {
        rx: oneshot::Receiver<T>,
        waiter_id: u64,
    },
}

impl<T> Default for Mailbox<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn mailbox_key_orders_as_src_ctx_tag_tuple() {
        let key: MailboxKey = (1, 2, 3);
        let (src, ctx, tag) = key;
        assert_eq!((src, ctx, tag), (1u32, 2u64, 3u64));
    }

    #[test]
    fn new_mailbox_compiles_for_a_representative_payload_type() {
        let _mailbox: Mailbox<Vec<f64>> = Mailbox::new();
    }

    #[test]
    fn push_then_try_pop_is_fifo_per_key() {
        let mb: Mailbox<u32> = Mailbox::new();
        let key = (1, 0, 7);
        for value in 0..4 {
            mb.push(key, value).expect("push");
        }
        assert_eq!(mb.len(key).expect("len"), 4);
        for expected in 0..4 {
            assert_eq!(mb.try_pop(key).expect("try_pop"), Some(expected));
        }
        assert_eq!(mb.try_pop(key).expect("try_pop"), None);
    }

    #[test]
    fn distinct_keys_do_not_interfere() {
        let mb: Mailbox<&'static str> = Mailbox::new();
        mb.push((1, 0, 1), "one").expect("push");
        mb.push((1, 0, 2), "two").expect("push");
        mb.push((2, 0, 1), "other-src").expect("push");
        mb.push((1, 9, 1), "other-ctx").expect("push");

        assert_eq!(mb.try_pop((1, 0, 2)).expect("try_pop"), Some("two"));
        assert_eq!(mb.try_pop((1, 0, 1)).expect("try_pop"), Some("one"));
        assert_eq!(mb.try_pop((2, 0, 1)).expect("try_pop"), Some("other-src"));
        assert_eq!(mb.try_pop((1, 9, 1)).expect("try_pop"), Some("other-ctx"));
    }

    #[test]
    fn over_capacity_push_errors_rather_than_blocking() {
        let mb: Mailbox<u8> = Mailbox::with_key_capacity(2);
        let key = (0, 0, 0);
        mb.push(key, 1).expect("push");
        mb.push(key, 2).expect("push");
        let err = mb.push(key, 3).expect_err("third push exceeds capacity");
        assert!(matches!(
            err,
            NetError::MailboxFull {
                src: 0,
                ctx: 0,
                tag: 0,
                capacity: 2
            }
        ));
        // The mailbox is still usable and lost nothing.
        assert_eq!(mb.try_pop(key).expect("try_pop"), Some(1));
    }

    #[tokio::test]
    async fn pop_wakes_on_a_later_push() {
        let mb: Arc<Mailbox<u32>> = Arc::new(Mailbox::new());
        let key = (3, 1, 4);
        let reader = {
            let mb = Arc::clone(&mb);
            tokio::spawn(async move { mb.pop(key).await })
        };
        // Give the reader a chance to park before the push arrives.
        tokio::task::yield_now().await;
        mb.push(key, 99).expect("push");
        let got = reader.await.expect("join").expect("pop");
        assert_eq!(got, 99);
    }

    #[tokio::test]
    async fn pop_timeout_names_the_missing_key() {
        let mb: Mailbox<u32> = Mailbox::new();
        let err = mb
            .pop_timeout((5, 6, 7), Duration::from_millis(20))
            .await
            .expect_err("nothing was ever sent");
        match err {
            NetError::RecvTimeout { src, ctx, tag, .. } => {
                assert_eq!((src, ctx, tag), (5, 6, 7));
            }
            other => panic!("expected RecvTimeout, got {other:?}"),
        }
        assert!(err.to_string().contains("src 5"));
    }

    #[tokio::test]
    async fn timed_out_waiter_leaves_no_leak_and_loses_no_message() {
        let mb: Mailbox<u32> = Mailbox::new();
        let key = (1, 1, 1);
        let _ = mb
            .pop_timeout(key, Duration::from_millis(10))
            .await
            .expect_err("times out");
        // The abandoned waiter must have been unregistered, so a later push
        // queues normally instead of vanishing into a dead oneshot.
        assert_eq!(mb.live_keys(), 0);
        mb.push(key, 7).expect("push");
        assert_eq!(mb.try_pop(key).expect("try_pop"), Some(7));
    }

    #[tokio::test]
    async fn push_racing_an_expiring_waiter_never_drops_the_payload() {
        // Hammer the exact interleaving where the timeout fires around the
        // same instant as the delivery: whoever wins, the payload must end
        // up either returned to the popper or still queued — never lost.
        for _ in 0..64 {
            let mb: Arc<Mailbox<u32>> = Arc::new(Mailbox::new());
            let key = (0, 0, 0);
            let popper = {
                let mb = Arc::clone(&mb);
                tokio::spawn(async move { mb.pop_timeout(key, Duration::from_millis(1)).await })
            };
            tokio::time::sleep(Duration::from_millis(1)).await;
            let pushed = mb.push(key, 123).is_ok();
            assert!(pushed, "push must not fail");
            let popped = popper.await.expect("join");
            match popped {
                Ok(value) => assert_eq!(value, 123),
                Err(NetError::RecvTimeout { .. }) => {
                    assert_eq!(
                        mb.try_pop(key).expect("try_pop"),
                        Some(123),
                        "a timed-out pop must leave the payload queued"
                    );
                }
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn close_wakes_parked_waiters() {
        let mb: Arc<Mailbox<u32>> = Arc::new(Mailbox::new());
        let key = (0, 0, 0);
        let reader = {
            let mb = Arc::clone(&mb);
            tokio::spawn(async move { mb.pop(key).await })
        };
        tokio::task::yield_now().await;
        mb.close();
        let err = reader.await.expect("join").expect_err("closed");
        assert!(matches!(err, NetError::MailboxClosed));
        assert!(matches!(mb.push(key, 1), Err(NetError::MailboxClosed)));
        assert!(mb.is_closed());
    }

    #[tokio::test]
    async fn two_waiters_on_one_key_are_served_fifo() {
        let mb: Arc<Mailbox<u32>> = Arc::new(Mailbox::new());
        let key = (2, 2, 2);
        let first = {
            let mb = Arc::clone(&mb);
            tokio::spawn(async move { mb.pop(key).await })
        };
        tokio::task::yield_now().await;
        let second = {
            let mb = Arc::clone(&mb);
            tokio::spawn(async move { mb.pop(key).await })
        };
        tokio::task::yield_now().await;

        mb.push(key, 10).expect("push");
        mb.push(key, 20).expect("push");
        assert_eq!(first.await.expect("join").expect("pop"), 10);
        assert_eq!(second.await.expect("join").expect("pop"), 20);
    }
}
