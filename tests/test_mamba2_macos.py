import unittest

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS


class TestMamba2MacOS(unittest.TestCase):
    """Tests for the macOS-compatible Mamba2 implementation."""
    
    def test_basic_forward(self):
        """Test basic forward pass of Mamba2MacOS."""
        # Model parameters
        d_model = 64
        d_state = 16
        batch_size = 2
        seq_len = 8
        
        # Create model
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        out = model(x)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
    
    def test_inference_cache(self):
        """Test inference cache allocation and usage."""
        # Model parameters
        d_model = 64
        d_state = 16
        batch_size = 2
        seq_len = 8
        
        # Create model
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
            layer_idx=0  # Required for inference cache
        )
        
        # Allocate cache
        conv_state, ssm_state = model.allocate_inference_cache(batch_size, seq_len)
        
        # Check cache shapes
        self.assertEqual(conv_state.shape, (batch_size, model.d_inner, model.d_conv))
        self.assertEqual(ssm_state.shape, (batch_size, model.nheads, d_state))
    
    def test_step_function(self):
        """Test the step function for autoregressive generation."""
        # Model parameters
        d_model = 64
        d_state = 16
        batch_size = 2
        
        # Create model
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        # Create input (single token)
        x = torch.randn(batch_size, 1, d_model)
        
        # Allocate states
        conv_state, ssm_state = model.allocate_inference_cache(batch_size, 1)
        
        # Step function
        out, new_conv_state, new_ssm_state = model.step(x, conv_state, ssm_state)
        
        # Check shapes
        self.assertEqual(out.shape, (batch_size, 1, d_model))
        self.assertEqual(new_conv_state.shape, conv_state.shape)
        self.assertEqual(new_ssm_state.shape, ssm_state.shape)
    
    def test_variable_length(self):
        """Test handling of variable length sequences."""
        # Model parameters
        d_model = 64
        d_state = 16
        batch_size = 2
        seq_lens = [5, 8]  # Different sequence lengths
        
        # Create model
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        # Create inputs with different lengths
        x1 = torch.randn(batch_size, seq_lens[0], d_model)
        x2 = torch.randn(batch_size, seq_lens[1], d_model)
        
        # Forward pass
        out1 = model(x1)
        out2 = model(x2)
        
        # Check output shapes
        self.assertEqual(out1.shape, (batch_size, seq_lens[0], d_model))
        self.assertEqual(out2.shape, (batch_size, seq_lens[1], d_model))
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        # Model parameters
        d_model = 64
        d_state = 16
        batch_size = 2
        seq_len = 8
        
        # Create model
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        # Forward pass
        out = model(x)
        
        # Create dummy target
        target = torch.randn_like(out)
        
        # Compute loss and backward pass
        loss = nn.MSELoss()(out, target)
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.in_proj.weight.grad)
        self.assertIsNotNone(model.out_proj.weight.grad)
        self.assertIsNotNone(model.conv1d.weight.grad)
        self.assertIsNotNone(model.A_log.grad)
        self.assertIsNotNone(model.D.grad)
        self.assertIsNotNone(model.dt_bias.grad) 