import sys
from importlib.metadata import version

import torch
from einops import rearrange, reduce, repeat

# Add a print statement to explain what's happening
print("Mamba SSM on macOS Apple Silicon - Minimal Demo")
print("----------------------------------------------")
print("Note: Using reference implementations as CUDA/Triton extensions are not supported on Apple Silicon")

# Import version directly from the file to avoid problematic imports
sys.path.insert(0, '.')
exec(open('mamba_ssm/__init__.py').readlines()[0])
print(f"Mamba SSM version: {__version__}")

# Get version using importlib.metadata
mamba_ssm_version = version('mamba-ssm')
print(f"Mamba SSM version: {mamba_ssm_version}")

# Define a minimal SSM reference implementation
def simple_ssm_scan(x, A, B, C, D=None):
    """
    Simple SSM scan implementation without requiring CUDA/Triton
    
    Args:
        x: input tensor of shape (batch, seq_len, dim)
        A: state matrix of shape (d_state, d_state)
        B: input matrix of shape (dim, d_state)
        C: output matrix of shape (d_state, dim)
        D: skip connection parameter of shape (dim,)
    """
    batch, seq_len, dim = x.shape
    d_state = A.shape[0]
    h = torch.zeros((batch, d_state), dtype=x.dtype)
    output = torch.zeros_like(x)
    
    # Simple sequential scan
    for t in range(seq_len):
        # Update hidden state: h = Ah + Bx
        # Reshape for matrix multiplication and preserve batch dimension
        h_expanded = rearrange(h, 'b d -> b 1 d')
        x_t = rearrange(x[:, t], 'b d -> b 1 d')
        
        # Matrix multiplications
        h = rearrange(torch.matmul(h_expanded, A), 'b 1 d -> b d') + \
            rearrange(torch.matmul(x_t, B), 'b 1 d -> b d')
            
        # Compute output: y = Ch + Dx
        y = rearrange(torch.matmul(rearrange(h, 'b d -> b 1 d'), C), 'b 1 d -> b d')
        
        if D is not None:
            # Element-wise multiplication with broadcasting
            y = y + x[:, t] * D
            
        output[:, t] = y
    
    return output

# Create sample input
batch_size = 2
seq_len = 10
hidden_dim = 8
d_state = 4

# Create input and SSM parameters with correct shapes
x = torch.randn(batch_size, seq_len, hidden_dim)
A = torch.randn(d_state, d_state)  # State transition matrix
B = torch.randn(hidden_dim, d_state)  # Input projection
C = torch.randn(d_state, hidden_dim)  # Output projection
D = torch.ones(hidden_dim)  # Skip connection

print(f"\nRunning simple SSM scan with shapes:")
print(f"  Input: {x.shape}")
print(f"  A: {A.shape}")
print(f"  B: {B.shape}")
print(f"  C: {C.shape}")
print(f"  D: {D.shape}")

# Run the simplified SSM
output = simple_ssm_scan(x, A, B, C, D)
print(f"\nOutput shape: {output.shape}")
print(f"First few values of output[0,0]: {output[0, 0, :4].tolist()}")

print("\nSimple Mamba SSM demonstration completed successfully!") 