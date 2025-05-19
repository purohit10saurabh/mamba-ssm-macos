import math
import sys

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat

print("Mamba SSM on macOS Apple Silicon - Selective Scan Demo")
print("-----------------------------------------------------")
print("Note: Using reference implementations as CUDA/Triton extensions are not supported on Apple Silicon")

# Import version directly from the file to avoid problematic imports
sys.path.insert(0, '.')
exec(open('mamba_ssm/__init__.py').readlines()[0])
print(f"Mamba SSM version: {__version__}")

# Define a selective SSM reference implementation based on the Mamba paper
def selective_scan_ref(u, delta, A, B, C, D=None):
    """
    Selective SSM scan based on Mamba paper
    
    Args:
        u: input tensor of shape (batch, seq_len, dim)
        delta: timescale parameter of shape (batch, seq_len, dim)
        A: state matrix of shape (dim, d_state)
        B: input matrix of shape (batch, seq_len, dim, d_state)
        C: output matrix of shape (batch, seq_len, dim, d_state)
        D: skip connection parameter of shape (dim,)
    """
    batch, seq_len, dim = u.shape
    _, _, _, d_state = B.shape
    
    # Discretize A using delta: Abar = exp(A * delta)
    # For simplicity, we assume A is diagonal, so we just use exp(A*delta)
    # In the actual Mamba implementation, this is more complex for non-diagonal A
    delta = rearrange(delta, 'b s d -> b s d 1')
    A = rearrange(A, 'd s -> 1 1 d s')
    Abar = torch.exp(A * delta)
    
    # Initialize state
    h = torch.zeros((batch, dim, d_state), dtype=u.dtype)
    ys = []
    
    # Scan through sequence
    for t in range(seq_len):
        # Extract current inputs and parameters
        u_t = u[:, t]  # (batch, dim)
        Abar_t = Abar[:, t]  # (batch, dim, d_state)
        B_t = B[:, t]  # (batch, dim, d_state)
        C_t = C[:, t]  # (batch, dim, d_state)
        
        # State update: h_t = A_t * h_{t-1} + B_t * u_t
        u_expanded = rearrange(u_t, 'b d -> b d 1')
        h = Abar_t * h + B_t * u_expanded
        
        # Output: y_t = C_t * h_t + D * u_t
        y_t = reduce(C_t * h, 'b d s -> b d', 'sum')
        if D is not None:
            y_t = y_t + D * u_t
            
        ys.append(y_t)
    
    return rearrange(ys, 't b d -> b t d')  # (batch, seq_len, dim)

# Create a simple selective SSM layer similar to what's used in Mamba
class SimpleSelectiveSSM(torch.nn.Module):
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Parameter matrices
        # S4 part
        self.A = torch.nn.Parameter(torch.randn(d_model, d_state))
        self.D = torch.nn.Parameter(torch.ones(d_model))
        
        # Calculate output size for the linear projection to match tensor shapes
        self.proj_size = 2*d_model + 2*d_model*d_state
        self.in_proj = torch.nn.Linear(d_model, self.proj_size)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        self._init_parameters()
        
    def _init_parameters(self):
        # Initialize parameters similarly to Mamba
        # A initialization - place poles in a specific range
        dl = torch.arange(self.d_state).float() / self.d_state
        A_real = -torch.exp(math.log(1000) * dl)
        # Just set the same values for each dimension - simplification for demo
        self.A.data = repeat(A_real, 's -> d s', d=self.d_model)
        
        # B, C will be computed on-the-fly based on input
        
        # Initialize projection weight
        torch.nn.init.xavier_uniform_(self.in_proj.weight)
        torch.nn.init.zeros_(self.in_proj.bias)
        
    def forward(self, x):
        batch, seq_len, dim = x.shape
        assert dim == self.d_model, f"Input dimension {dim} doesn't match model dimension {self.d_model}"
        
        # Project input to get parameters
        x_proj = self.in_proj(x)  # (batch, seq_len, self.proj_size)
        
        # Calculate chunk sizes
        d_model, d_state = self.d_model, self.d_state
        chunk_sizes = [d_model, d_model, d_model*d_state, d_model*d_state]
        
        # Split projection into components
        u, delta, B_proj, C_proj = torch.split(x_proj, chunk_sizes, dim=-1)
        
        # Activation for delta (timescale)
        delta = F.softplus(delta)  # ensure positive
        
        # Reshape B_proj and C_proj using einops
        B = rearrange(B_proj, 'b s (d ds) -> b s d ds', d=d_model, ds=d_state)
        C = rearrange(C_proj, 'b s (d ds) -> b s d ds', d=d_model, ds=d_state)
        
        # Apply selective scan
        y = selective_scan_ref(u, delta, self.A, B, C, self.D)
        
        # Apply dropout and return
        return self.dropout(y)

# Try the simple Mamba-like model
d_model = 16
d_state = 8
batch_size = 2
seq_len = 10

# Create model and input
model = SimpleSelectiveSSM(d_model, d_state)
x = torch.randn(batch_size, seq_len, d_model)

print(f"\nRunning selective scan with:")
print(f"  Model dimensions: d_model={d_model}, d_state={d_state}")
print(f"  Input shape: {x.shape}")

# Run model
with torch.no_grad():
    output = model(x)

print(f"\nOutput shape: {output.shape}")
print(f"First few values of output[0,0]: {output[0, 0, :4].tolist()}")

print("\nSelective SSM scan demonstration completed successfully!") 