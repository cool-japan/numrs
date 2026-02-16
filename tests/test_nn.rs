//! Comprehensive Neural Network Tests
//!
//! This integration test module tests all neural network primitives.

mod nn {
    mod test_activation {
        include!("nn/test_activation.rs");
    }

    mod test_normalization {
        include!("nn/test_normalization.rs");
    }

    mod test_conv {
        include!("nn/test_conv.rs");
    }

    mod test_pooling {
        include!("nn/test_pooling.rs");
    }

    mod test_loss {
        include!("nn/test_loss.rs");
    }

    mod test_simd_ops {
        include!("nn/test_simd_ops.rs");
    }
}
