import unittest

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS


class TestMamba2MacOS(unittest.TestCase):
    def test_basic_forward(self):
        d_model, d_state, batch_size, seq_len = 64, 16, 2, 8
        
        model = Mamba2MacOS(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        x = torch.randn(batch_size, seq_len, d_model)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
    
    def test_shape_consistency_large_model(self):
        d_model, d_state, headdim, batch_size, seq_len = 256, 64, 64, 2, 32
        
        model = Mamba2MacOS(d_model=d_model, d_state=d_state, headdim=headdim, d_conv=4, expand=2)
        x = torch.randn(batch_size, seq_len, d_model)
        out = model(x)
        
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
        self.assertEqual(model.d_inner, 512)
        self.assertEqual(model.nheads, 8)
        self.assertEqual(model.headdim, 64)
    
    def test_tensor_shape_flow(self):
        """Test intermediate tensor shapes throughout forward pass."""
        d_model = 128
        d_state = 32
        headdim = 32
        batch_size = 2
        seq_len = 16
        
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            d_conv=4,
            expand=2
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Hook to capture intermediate shapes
        shapes = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    shapes[name] = output.shape
                elif isinstance(output, tuple):
                    shapes[name] = tuple(o.shape if isinstance(o, torch.Tensor) else str(o) for o in output)
            return hook
        
        # Register hooks
        model.in_proj.register_forward_hook(hook_fn('in_proj'))
        model.conv1d.register_forward_hook(hook_fn('conv1d'))
        model.dt_proj.register_forward_hook(hook_fn('dt_proj'))
        model.norm.register_forward_hook(hook_fn('norm'))
        model.out_proj.register_forward_hook(hook_fn('out_proj'))
        
        # Forward pass
        out = model(x)
        
        # Verify key intermediate shapes
        self.assertEqual(shapes['in_proj'], (batch_size, seq_len, model.d_inner * 2))
        # Note: conv1d adds padding (d_conv-1=3), so output is seq_len + 3 = 19
        # but it gets truncated back to seq_len in the forward pass
        self.assertEqual(shapes['conv1d'], (batch_size, model.d_inner, seq_len + model.d_conv - 1))
        self.assertEqual(shapes['norm'], (batch_size, seq_len, model.d_inner))
        self.assertEqual(shapes['out_proj'], (batch_size, seq_len, d_model))
    
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
    
    def test_step_function_large_model(self):
        """Test step function with larger model configuration."""
        d_model = 256
        d_state = 64
        headdim = 64
        batch_size = 2
        
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            d_conv=4,
            expand=2
        )
        
        x = torch.randn(batch_size, 1, d_model)
        conv_state, ssm_state = model.allocate_inference_cache(batch_size, 1)
        
        # Step function with larger model
        out, new_conv_state, new_ssm_state = model.step(x, conv_state, ssm_state)
        
        self.assertEqual(out.shape, (batch_size, 1, d_model))
        self.assertEqual(new_conv_state.shape, (batch_size, model.d_inner, model.d_conv))
        self.assertEqual(new_ssm_state.shape, (batch_size, model.nheads, d_state))
    
    def test_multiple_step_consistency(self):
        """Test that multiple step calls maintain consistency."""
        d_model = 128
        d_state = 32
        headdim = 32
        batch_size = 2
        num_steps = 5
        
        model = Mamba2MacOS(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            d_conv=4,
            expand=2
        )
        
        conv_state, ssm_state = model.allocate_inference_cache(batch_size, 1)
        outputs = []
        
        # Run multiple steps
        for i in range(num_steps):
            x = torch.randn(batch_size, 1, d_model)
            out, conv_state, ssm_state = model.step(x, conv_state, ssm_state)
            outputs.append(out)
            
            # Verify shapes remain consistent
            self.assertEqual(out.shape, (batch_size, 1, d_model))
            self.assertEqual(conv_state.shape, (batch_size, model.d_inner, model.d_conv))
            self.assertEqual(ssm_state.shape, (batch_size, model.nheads, d_state))
        
        # Verify we got all outputs
        self.assertEqual(len(outputs), num_steps)
    
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
    
    def test_different_headdim_configs(self):
        """Test different headdim configurations."""
        d_model = 192
        d_state = 48
        batch_size = 2
        seq_len = 16
        
        # Test different headdim values
        headdims = [32, 48, 64]
        
        for headdim in headdims:
            with self.subTest(headdim=headdim):
                model = Mamba2MacOS(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    d_conv=4,
                    expand=2
                )
                
                # Verify nheads calculation
                expected_nheads = model.d_inner // headdim
                self.assertEqual(model.nheads, expected_nheads)
                
                # Test forward pass
                x = torch.randn(batch_size, seq_len, d_model)
                out = model(x)
                self.assertEqual(out.shape, (batch_size, seq_len, d_model))
    
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
    
    def test_edge_case_dimensions(self):
        """Test edge cases with unusual dimension configurations."""
        test_configs = [
            {"d_model": 32, "d_state": 8, "headdim": 16},   # Small model
            {"d_model": 512, "d_state": 128, "headdim": 128}, # Large model
            {"d_model": 96, "d_state": 24, "headdim": 24},   # Non-power-of-2
        ]
        
        batch_size = 1
        seq_len = 4
        
        for config in test_configs:
            with self.subTest(**config):
                model = Mamba2MacOS(
                    d_model=config["d_model"],
                    d_state=config["d_state"],
                    headdim=config["headdim"],
                    d_conv=4,
                    expand=2
                )
                
                x = torch.randn(batch_size, seq_len, config["d_model"])
                
                # Test forward pass with edge case dimensions
                out = model(x)
                self.assertEqual(out.shape, (batch_size, seq_len, config["d_model"]))


def run_mamba2_tests():
    print("🧪 Running Mamba2 macOS tests...")
    
    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMamba2MacOS)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ All Mamba2 macOS tests passed")
            return True
        else:
            print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
            return False
            
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        return False

if __name__ == "__main__":
    run_mamba2_tests() 