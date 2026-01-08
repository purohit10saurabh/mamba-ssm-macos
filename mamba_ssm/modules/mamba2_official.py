import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:

    class RMSNormGated(nn.Module):
        def __init__(
            self, d, eps=1e-5, norm_before_gate=False, device=None, dtype=None
        ):
            super().__init__()
            self.eps = eps
            self.norm_before_gate = norm_before_gate
            self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

        def forward(self, x, z=None):
            if z is not None and not self.norm_before_gate:
                x = x * F.silu(z)
            x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x_norm = x_norm * self.weight
            if z is not None and self.norm_before_gate:
                x_norm = x_norm * F.silu(z)
            return x_norm


try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    USE_OPTIMIZED_KERNELS = True
except ImportError:
    USE_OPTIMIZED_KERNELS = False

    def mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=256,
        D=None,
        z=None,
        seq_idx=None,
        initial_states=None,
        **kwargs,
    ):
        batch, seqlen, nheads, headdim = x.shape
        ngroups = B.shape[2] if B.ndim > 3 else 1

        if ngroups < nheads:
            heads_per_group = nheads // ngroups
            B = repeat(B, "b l g d -> b l (g h) d", h=heads_per_group)
            C = repeat(C, "b l g d -> b l (g h) d", h=heads_per_group)

        dA = torch.einsum("blh,h->blh", dt, A)
        dB_u = torch.einsum("blh,blhp,blhn->blhpn", dt, x, B)

        dA_padded = F.pad(dA[:, 1:], (0, 0, 0, 1))
        dA_cumsum = dA_padded.flip(1).cumsum(1).flip(1)
        dA_cumsum_exp = torch.exp(dA_cumsum)

        dA_cumsum_expanded = dA_cumsum_exp.unsqueeze(-1).unsqueeze(-1)
        x_scaled = dB_u * dA_cumsum_expanded
        x_cumsum = x_scaled.cumsum(1)
        states = x_cumsum / (dA_cumsum_expanded + 1e-12)

        y = torch.einsum("blhpn,blhn->blhp", states, C)

        if D is not None:
            y = y + x * D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        return y

    def mamba_split_conv1d_scan_combined(*args, **kwargs):
        raise NotImplementedError(
            "Reference implementation not available for split conv1d scan"
        )


class Mamba2Official(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model, self.d_state, self.d_conv, self.conv_init, self.expand = (
            d_model,
            d_state,
            d_conv,
            conv_init,
            expand,
        )
        self.d_inner, self.headdim, self.ngroups = expand * d_model, headdim, ngroups

        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        self.dt_limit, self.learnable_init_states, self.activation, self.chunk_size = (
            dt_limit,
            learnable_init_states,
            activation,
            chunk_size,
        )
        self.use_mem_eff_path, self.layer_idx = (
            use_mem_eff_path and USE_OPTIMIZED_KERNELS,
            layer_idx,
        )

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs
        )
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, u, seq_idx=None, inference_params=None, **kwargs):
        batch, seqlen, dim = u.shape
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        initial_states = (
            repeat(self.init_states, "... -> b ...", b=batch)
            if self.learnable_init_states
            else None
        )
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        if self.use_mem_eff_path:
            try:
                return mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=False,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
            except Exception:
                pass

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)

        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))[
                :, :seqlen, :
            ]
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            ).transpose(1, 2)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )

        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        return self.out_proj(y)
