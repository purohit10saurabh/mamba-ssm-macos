import unittest

import torch
import torch.nn as nn

import mamba_ssm
from mamba_ssm.modules.mamba_simple import Mamba


class MambaClassifier(nn.Module):
    """A simple classifier using Mamba for sequence modeling."""
    
    def __init__(self, vocab_size=1000, d_model=64, d_state=16, n_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        x = self.mamba(x)      # [batch_size, seq_len, d_model]
        
        # Use last token for classification
        x = x[:, -1]           # [batch_size, d_model]
        x = self.classifier(x) # [batch_size, n_classes]
        return x


class TestMambaMacOS(unittest.TestCase):
    """Tests for Mamba running on macOS Apple Silicon."""
    
    def test_basic_forward(self):
        """Test a basic Mamba model forward pass."""
        # Create a small model for testing
        d_model = 16
        d_state = 4
        batch_size = 2
        seq_len = 6

        # Initialize model
        model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=2,
            expand=2
        )

        # Create input
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward pass
        out = model(x)

        # Check output shape
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_classifier_forward_backward(self):
        """Test a more complex model with embedding, classification, and backprop."""
        # Model parameters
        vocab_size = 1000
        d_model = 64
        d_state = 16
        n_classes = 10
        batch_size = 8
        seq_len = 32

        # Initialize model
        model = MambaClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=d_state,
            n_classes=n_classes
        )

        # Create dummy input (token IDs)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        out = model(x)

        # Check output shape
        self.assertEqual(out.shape, (batch_size, n_classes))

        # Test backpropagation
        target = torch.randint(0, n_classes, (batch_size,))
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, target)
        
        # This should not raise an error
        loss.backward()


def run_mamba_tests():
    print("🧪 Running Mamba macOS tests...")
    
    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMambaMacOS)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ All Mamba macOS tests passed")
            return True
        else:
            print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
            return False
            
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        return False

if __name__ == "__main__":
    run_mamba_tests() 