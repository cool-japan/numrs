//! Communication Layer for Distributed Computing
//!
//! This module provides low-level communication primitives for distributed computing,
//! implemented in Pure Rust using tokio for async networking and oxicode for serialization.
//!
//! # Features
//!
//! - Point-to-point communication (send/recv)
//! - Non-blocking async operations
//! - Message serialization with oxicode (COOLJAPAN policy)
//! - Connection pooling for efficiency
//! - Timeout handling
//! - Network failure recovery
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::comm::*;
//! use std::net::SocketAddr;
//!
//! # async fn example() -> Result<(), CommunicationError> {
//! let addr: SocketAddr = "127.0.0.1:5000".parse().expect("valid socket address literal");
//! let mut manager = ConnectionManager::new(addr).await?;
//!
//! // Send data to another process
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0];
//! let msg = Message::new(0, 1, 42, data)?;
//! manager.send(msg).await?;
//!
//! // Receive data
//! let received: Message<f64> = manager.recv().await?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, RwLock};
use tokio::time::timeout;

/// Errors that can occur during communication operations
#[derive(Error, Debug, Clone)]
pub enum CommunicationError {
    #[error("Network I/O error: {0}")]
    IoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Connection error to {addr}: {msg}")]
    ConnectionError { addr: String, msg: String },

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid message format: {0}")]
    InvalidMessage(String),

    #[error("Buffer overflow: tried to write {size} bytes, buffer capacity {capacity}")]
    BufferOverflow { size: usize, capacity: usize },

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Invalid rank: {0}")]
    InvalidRank(usize),
}

/// Message tag type for distinguishing different message types
pub type MessageTag = u32;

/// Message header containing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Source process rank
    pub source: usize,
    /// Destination process rank
    pub dest: usize,
    /// Message tag for identification
    pub tag: MessageTag,
    /// Size of payload in bytes
    pub payload_size: usize,
    /// Message sequence number
    pub sequence: u64,
}

impl MessageHeader {
    /// Create a new message header
    pub fn new(
        source: usize,
        dest: usize,
        tag: MessageTag,
        payload_size: usize,
        sequence: u64,
    ) -> Self {
        Self {
            source,
            dest,
            tag,
            payload_size,
            sequence,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, CommunicationError> {
        let config = oxicode::config::standard();
        oxicode::serde::encode_to_vec(self, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CommunicationError> {
        let config = oxicode::config::standard();
        let (header, _): (Self, usize) = oxicode::serde::decode_from_slice(bytes, config)
            .map_err(|e| CommunicationError::DeserializationError(e.to_string()))?;
        Ok(header)
    }
}

/// A message containing data to be sent between processes
#[derive(Debug, Clone)]
pub struct Message<T> {
    /// Message header with metadata
    pub header: MessageHeader,
    /// Message payload
    pub payload: Vec<T>,
}

impl<T: Serialize + for<'de> Deserialize<'de> + Clone> Message<T> {
    /// Create a new message
    pub fn new(
        source: usize,
        dest: usize,
        tag: MessageTag,
        payload: Vec<T>,
    ) -> Result<Self, CommunicationError> {
        // Serialize payload to get size
        let config = oxicode::config::standard();
        let payload_bytes = oxicode::serde::encode_to_vec(&payload, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))?;

        let header = MessageHeader::new(source, dest, tag, payload_bytes.len(), 0);

        Ok(Self { header, payload })
    }

    /// Create a message with a specific sequence number
    pub fn with_sequence(
        source: usize,
        dest: usize,
        tag: MessageTag,
        payload: Vec<T>,
        sequence: u64,
    ) -> Result<Self, CommunicationError> {
        let config = oxicode::config::standard();
        let payload_bytes = oxicode::serde::encode_to_vec(&payload, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))?;

        let header = MessageHeader::new(source, dest, tag, payload_bytes.len(), sequence);

        Ok(Self { header, payload })
    }

    /// Serialize message to bytes (header + payload)
    pub fn to_bytes(&self) -> Result<Vec<u8>, CommunicationError> {
        let header_bytes = self.header.to_bytes()?;
        let config = oxicode::config::standard();
        let payload_bytes = oxicode::serde::encode_to_vec(&self.payload, config)
            .map_err(|e| CommunicationError::SerializationError(e.to_string()))?;

        // Format: [header_size: u32][header][payload]
        let header_size = header_bytes.len() as u32;
        let mut bytes = Vec::with_capacity(4 + header_bytes.len() + payload_bytes.len());
        bytes.extend_from_slice(&header_size.to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload_bytes);

        Ok(bytes)
    }

    /// Deserialize message from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CommunicationError> {
        if bytes.len() < 4 {
            return Err(CommunicationError::InvalidMessage(
                "Insufficient bytes for header size".to_string(),
            ));
        }

        // Read header size
        let header_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        if bytes.len() < 4 + header_size {
            return Err(CommunicationError::InvalidMessage(format!(
                "Insufficient bytes for header: expected {}, got {}",
                header_size,
                bytes.len() - 4
            )));
        }

        // Read header
        let header = MessageHeader::from_bytes(&bytes[4..4 + header_size])?;

        // Read payload
        let payload_bytes = &bytes[4 + header_size..];
        let config = oxicode::config::standard();
        let (payload, _): (Vec<T>, usize) =
            oxicode::serde::decode_from_slice(payload_bytes, config)
                .map_err(|e| CommunicationError::DeserializationError(e.to_string()))?;

        Ok(Self { header, payload })
    }

    /// Get the source rank
    pub fn source(&self) -> usize {
        self.header.source
    }

    /// Get the destination rank
    pub fn dest(&self) -> usize {
        self.header.dest
    }

    /// Get the message tag
    pub fn tag(&self) -> MessageTag {
        self.header.tag
    }

    /// Get the sequence number
    pub fn sequence(&self) -> u64 {
        self.header.sequence
    }
}

/// Communication channel for sending and receiving messages
pub struct CommunicationChannel {
    /// TCP stream for this connection
    stream: Arc<Mutex<TcpStream>>,
    /// Remote address
    remote_addr: SocketAddr,
    /// Sequence counter for messages
    sequence: Arc<Mutex<u64>>,
}

impl CommunicationChannel {
    /// Create a new communication channel
    pub fn new(stream: TcpStream, remote_addr: SocketAddr) -> Self {
        Self {
            stream: Arc::new(Mutex::new(stream)),
            remote_addr,
            sequence: Arc::new(Mutex::new(0)),
        }
    }

    /// Send a message through this channel
    pub async fn send<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        mut message: Message<T>,
    ) -> Result<(), CommunicationError> {
        // Update sequence number
        let mut seq = self.sequence.lock().await;
        message.header.sequence = *seq;
        *seq += 1;
        drop(seq);

        // Serialize message
        let bytes = message.to_bytes()?;

        // Send message size first
        let size = bytes.len() as u64;
        let mut stream = self.stream.lock().await;

        stream
            .write_all(&size.to_le_bytes())
            .await
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;

        // Send message data
        stream
            .write_all(&bytes)
            .await
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;

        stream
            .flush()
            .await
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Receive a message from this channel
    pub async fn recv<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
    ) -> Result<Message<T>, CommunicationError> {
        let mut stream = self.stream.lock().await;

        // Read message size
        let mut size_bytes = [0u8; 8];
        stream.read_exact(&mut size_bytes).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                CommunicationError::ConnectionClosed
            } else {
                CommunicationError::IoError(e.to_string())
            }
        })?;

        let size = u64::from_le_bytes(size_bytes) as usize;

        // Read message data
        let mut bytes = vec![0u8; size];
        stream
            .read_exact(&mut bytes)
            .await
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;

        // Deserialize message
        Message::from_bytes(&bytes)
    }

    /// Receive a message with timeout
    pub async fn recv_timeout<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        duration: Duration,
    ) -> Result<Message<T>, CommunicationError> {
        timeout(duration, self.recv())
            .await
            .map_err(|_| CommunicationError::Timeout(duration))?
    }

    /// Get the remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }
}

/// Connection manager for managing multiple connections
pub struct ConnectionManager {
    /// Local bind address
    local_addr: SocketAddr,
    /// TCP listener for incoming connections
    listener: Arc<Mutex<TcpListener>>,
    /// Connection pool: rank -> channel
    connections: Arc<RwLock<HashMap<usize, Arc<CommunicationChannel>>>>,
    /// Mapping of addresses to ranks
    rank_addresses: Arc<RwLock<HashMap<usize, SocketAddr>>>,
    /// Next sequence number for messages
    next_sequence: Arc<Mutex<u64>>,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub async fn new(bind_addr: SocketAddr) -> Result<Self, CommunicationError> {
        let listener = TcpListener::bind(bind_addr).await.map_err(|e| {
            CommunicationError::ConnectionError {
                addr: bind_addr.to_string(),
                msg: e.to_string(),
            }
        })?;

        let local_addr = listener
            .local_addr()
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;

        Ok(Self {
            local_addr,
            listener: Arc::new(Mutex::new(listener)),
            connections: Arc::new(RwLock::new(HashMap::new())),
            rank_addresses: Arc::new(RwLock::new(HashMap::new())),
            next_sequence: Arc::new(Mutex::new(0)),
        })
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Register a rank with its address
    pub async fn register_rank(&self, rank: usize, addr: SocketAddr) {
        let mut addrs = self.rank_addresses.write().await;
        addrs.insert(rank, addr);
    }

    /// Get or create connection to a rank
    pub async fn get_connection(
        &self,
        rank: usize,
    ) -> Result<Arc<CommunicationChannel>, CommunicationError> {
        // Check if connection already exists
        {
            let conns = self.connections.read().await;
            if let Some(channel) = conns.get(&rank) {
                return Ok(Arc::clone(channel));
            }
        }

        // Get address for rank
        let addr = {
            let addrs = self.rank_addresses.read().await;
            addrs
                .get(&rank)
                .copied()
                .ok_or(CommunicationError::InvalidRank(rank))?
        };

        // Create new connection
        let stream =
            TcpStream::connect(addr)
                .await
                .map_err(|e| CommunicationError::ConnectionError {
                    addr: addr.to_string(),
                    msg: e.to_string(),
                })?;

        let channel = Arc::new(CommunicationChannel::new(stream, addr));

        // Store connection
        {
            let mut conns = self.connections.write().await;
            conns.insert(rank, Arc::clone(&channel));
        }

        Ok(channel)
    }

    /// Send a message to a specific rank
    pub async fn send<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        message: Message<T>,
    ) -> Result<(), CommunicationError> {
        let rank = message.dest();
        let channel = self.get_connection(rank).await?;
        channel.send(message).await
    }

    /// Receive a message (from any source)
    pub async fn recv<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
    ) -> Result<Message<T>, CommunicationError> {
        // Accept incoming connection
        let listener = self.listener.lock().await;
        let (stream, remote_addr) = listener
            .accept()
            .await
            .map_err(|e| CommunicationError::IoError(e.to_string()))?;
        drop(listener);

        // Create temporary channel for receiving
        let channel = CommunicationChannel::new(stream, remote_addr);
        channel.recv().await
    }

    /// Receive a message with timeout
    pub async fn recv_timeout<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &self,
        duration: Duration,
    ) -> Result<Message<T>, CommunicationError> {
        timeout(duration, self.recv())
            .await
            .map_err(|_| CommunicationError::Timeout(duration))?
    }

    /// Close all connections
    pub async fn close_all(&self) -> Result<(), CommunicationError> {
        let mut conns = self.connections.write().await;
        conns.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_header() {
        let header = MessageHeader::new(0, 1, 42, 1024, 100);
        assert_eq!(header.source, 0);
        assert_eq!(header.dest, 1);
        assert_eq!(header.tag, 42);
        assert_eq!(header.payload_size, 1024);
        assert_eq!(header.sequence, 100);
    }

    #[test]
    fn test_message_header_serialization() {
        let header = MessageHeader::new(0, 1, 42, 1024, 100);
        let bytes = header.to_bytes().expect("Serialization failed");
        let deserialized = MessageHeader::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(header.source, deserialized.source);
        assert_eq!(header.dest, deserialized.dest);
        assert_eq!(header.tag, deserialized.tag);
        assert_eq!(header.payload_size, deserialized.payload_size);
        assert_eq!(header.sequence, deserialized.sequence);
    }

    #[test]
    fn test_message_creation() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let msg = Message::new(0, 1, 42, data.clone()).expect("Message creation failed");

        assert_eq!(msg.source(), 0);
        assert_eq!(msg.dest(), 1);
        assert_eq!(msg.tag(), 42);
        assert_eq!(msg.payload, data);
    }

    #[test]
    fn test_message_serialization() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let msg = Message::new(0, 1, 42, data.clone()).expect("Message creation failed");

        let bytes = msg.to_bytes().expect("Serialization failed");
        let deserialized: Message<f64> =
            Message::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(msg.source(), deserialized.source());
        assert_eq!(msg.dest(), deserialized.dest());
        assert_eq!(msg.tag(), deserialized.tag());
        assert_eq!(msg.payload, deserialized.payload);
    }

    #[test]
    fn test_message_with_sequence() {
        let data = vec![1, 2, 3, 4];
        let msg = Message::with_sequence(0, 1, 42, data, 123).expect("Message creation failed");

        assert_eq!(msg.sequence(), 123);
    }

    #[tokio::test]
    async fn test_connection_manager_creation() {
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("Valid address");
        let manager = ConnectionManager::new(addr)
            .await
            .expect("Manager creation failed");

        // Just verify it was created successfully
        assert!(manager.local_addr().port() > 0);
    }

    #[tokio::test]
    async fn test_register_rank() {
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("Valid address");
        let manager = ConnectionManager::new(addr)
            .await
            .expect("Manager creation failed");

        let rank_addr: SocketAddr = "127.0.0.1:5001".parse().expect("Valid address");
        manager.register_rank(0, rank_addr).await;

        // Verify rank was registered
        let addrs = manager.rank_addresses.read().await;
        assert_eq!(addrs.get(&0), Some(&rank_addr));
    }
}
