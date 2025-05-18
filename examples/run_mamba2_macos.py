import sys
import warnings

import torch
import torch.nn as nn

print("Mamba2-like Model on macOS Apple Silicon - Compatibility Demo")
print("---------------------------------------------------------")
print("Note: Using custom implementation since Mamba2 requires Triton which is not supported on Apple Silicon")

# Import version directly from the file to avoid problematic imports
sys.path.insert(0, '.')
exec(open('mamba_ssm/__init__.py').readlines()[0])
print(f"Mamba SSM version: {__version__}")

from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba


class SimpleMamba2Like(nn.Module):
    """
    A simplified version of Mamba2 for macOS that uses only components available on Apple Silicon.
    This doesn't have all the Mamba2 features but provides a compatible interface.
    """
    
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dt_rank=None, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand
        self.d_inner = int(d_model * expand)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)  # for x and z
        
        # Use standard Mamba layer which works on macOS
        self.mamba_layer = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1  # Already expanded
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, d_model)
        batch, seq_len, _ = x.shape
        
        # Project input
        xz = self.in_proj(x)  # (batch_size, seq_len, d_inner*2)
        
        # Split into x and z streams
        x_and_z = rearrange(xz, "b l (c d) -> b l c d", c=2)
        x_proj, z_proj = x_and_z[:, :, 0], x_and_z[:, :, 1]  # Each: (batch_size, seq_len, d_inner)
        
        # Apply standard Mamba (which expects input as B D L)
        # x_proj = rearrange(x_proj, "b l d -> b d l")
        # z_proj = rearrange(z_proj, "b l d -> b d l")
        
        # Pass through Mamba layer
        output = self.mamba_layer(x_proj)  # (batch_size, d_inner, seq_len)
        
        # Apply SiLU gate
        output = output * torch.sigmoid(z_proj)
        
        # output = rearrange(output, "b l d -> b d l")
        
        # Project back to original dimension
        # import ipdb; ipdb.set_trace()
        output = self.out_proj(output)  # (batch_size, seq_len, d_model)
        
        return output


# Create sample input
batch_size = 4
seq_len = 16
d_model = 64

# Create model
model = SimpleMamba2Like(
    d_model=d_model,
    d_state=16,
    d_conv=4,
    expand=2
)

# Create input
x = torch.randn(batch_size, seq_len, d_model)

# Run inference
with torch.no_grad():
    output = model(x)

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"First few values of output[0,0]: {output[0, 0, :5].tolist()}")

print("\nSuccessfully ran Mamba2-like model on macOS Apple Silicon!") 