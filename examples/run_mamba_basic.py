import torch
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# Create sample input
batch_size = 2
seq_len = 10
hidden_dim = 16
input_tensor = torch.randn(batch_size, seq_len, hidden_dim)

# Create some sample SSM parameters
d_state = 4
A = torch.randn(hidden_dim, d_state)
B = torch.randn(batch_size, seq_len, hidden_dim, d_state)
C = torch.randn(batch_size, seq_len, hidden_dim, d_state)
D = torch.zeros(hidden_dim)

# Run the selective scan function (basic operation in Mamba)
print("Input shape:", input_tensor.shape)
output, _ = selective_scan_fn(
    input_tensor,
    A, B, C, D,
    is_variable_B=True, is_variable_C=True,
    delta_bias=None, delta_softplus=True
)
print("Output shape:", output.shape)
print("First few values of output:", output[0, 0, :5])

print("\nMamba SSM basic functionality demo completed successfully!") 