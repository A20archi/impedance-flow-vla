r"""
Compact conditional flow-matching action-chunk policy for the STAGE-1 MECHANISM GATE.
=====================================================================================
Two arms, IDENTICAL training objective (differentiable rollout-matching), differing ONLY in the
drift parameterization -- so any gap is the structural prior, not the objective:

  mode="vanilla"   : unstructured velocity field v_theta(a_tau, tau, c); Euler flow noise->action.
  mode="impedance" : the generated chunk is a 2nd-order semi-implicit mass-spring-damper toward a
                     predicted attractor a_goal(c), with SPD stiffness K and damping D from c
                     (mass M = I for the gate; M-shaping is a Stage-3 extension), plus residual f.
                     acc = -D w - K (a - a_goal) + f ;  w += ds*acc ;  a += ds*w   (validated in toy/).
                     K, D spectra are CAPPED for 10-step stability (toy: K < ~(1/dt)^2).

State-conditioned (full MetaWorld obs) for speed. Vision / VLM-predicted K,D is Stage 2-3.
"""
import torch, torch.nn as nn, torch.nn.functional as F


def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class CompactFlowPolicy(nn.Module):
    def __init__(self, cond_dim, act_dim=4, horizon=16, hidden=256, num_steps=10,
                 mode="vanilla", k_max=20.0, d_max=18.0, zeta_max=2.0):
        super().__init__()
        self.act_dim, self.H, self.num_steps, self.mode = act_dim, horizon, num_steps, mode
        self.k_max, self.d_max, self.zeta_max = k_max, d_max, zeta_max
        self.cond_enc = mlp([cond_dim, hidden, hidden])
        Dout = act_dim * horizon
        if mode == "vanilla":
            self.vnet = mlp([Dout + 1 + hidden, hidden, hidden, Dout])
        elif mode == "impedance":
            n = act_dim
            self.ntri = n * (n + 1) // 2
            self.kd_head = mlp([hidden, hidden, self.ntri + 1])      # K Cholesky params + zeta (>=1, critical damping)
            self.goal_head = mlp([hidden, hidden, Dout])             # attractor a_goal(c)
            self.fnet = mlp([2 * Dout + 1 + hidden, hidden, hidden, Dout])  # residual force f(a,w,s,c)
        else:
            raise ValueError(mode)

    # ---- SPD helpers (spectral cap for stability at few Euler steps; matrix sqrt for critical damping) ----
    def _chol_spd(self, params, n):
        B = params.shape[0]
        L = torch.zeros(B, n, n, device=params.device, dtype=params.dtype)
        ti = torch.tril_indices(n, n, device=params.device)
        L[:, ti[0], ti[1]] = params
        di = torch.arange(n, device=params.device)
        L[:, di, di] = F.softplus(L[:, di, di]) + 1e-3
        return L @ L.transpose(-1, -2)                               # SPD

    def _spectral_cap(self, A, cap):
        lam = torch.linalg.eigvalsh(A)[:, -1].clamp(min=cap)         # max eigenvalue
        return A * (cap / lam)[:, None, None]                        # cap spectral norm at `cap`

    def _sqrtm_spd(self, A):
        evals, evecs = torch.linalg.eigh(A)
        return (evecs * evals.clamp(min=1e-8).sqrt().unsqueeze(-2)) @ evecs.transpose(-1, -2)

    def forward(self, cond, noise):
        """cond [B, cond_dim], noise [B, H, act_dim] -> generated action chunk [B, H, act_dim]."""
        B = noise.shape[0]
        h = self.cond_enc(cond)
        if self.mode == "vanilla":
            dt = -1.0 / self.num_steps
            x = noise
            for step in range(self.num_steps):                       # SmolVLA convention: noise(tau=1)->data(tau=0)
                tau = 1.0 + step * dt
                inp = torch.cat([x.reshape(B, -1), x.new_full((B, 1), tau), h], dim=-1)
                v = self.vnet(inp).reshape(B, self.H, self.act_dim)
                x = x + dt * v
            return x
        else:
            n = self.act_dim
            kd = self.kd_head(h)
            K = self._spectral_cap(self._chol_spd(kd[:, :self.ntri], n), self.k_max)              # SPD, spectral<=k_max
            zeta = (1.0 + F.softplus(kd[:, self.ntri:self.ntri + 1])).clamp(max=self.zeta_max)    # [B,1] >=1: critically/over-damped => smooth
            D = self._spectral_cap(2.0 * zeta[:, :, None] * self._sqrtm_spd(K), self.d_max)       # D = 2 zeta sqrt(K)
            a_goal = self.goal_head(h).reshape(B, self.H, n)
            ds = 1.0 / self.num_steps
            a = noise; w = torch.zeros_like(noise)
            for step in range(self.num_steps):                       # 2nd-order semi-implicit damped spring noise->a_goal
                s = step * ds
                f = self.fnet(torch.cat([a.reshape(B, -1), w.reshape(B, -1), a.new_full((B, 1), s), h], dim=-1))
                f = f.reshape(B, self.H, n)
                Kx = torch.einsum("bij,bhj->bhi", K, a - a_goal)
                Dw = torch.einsum("bij,bhj->bhi", D, w)
                acc = -Dw - Kx + f
                w = w + ds * acc                                     # velocity first (semi-implicit -> stable)
                a = a + ds * w
            return a

    def loss(self, cond, expert_chunk, noise=None, smooth=1e-3):
        """rollout-match: MSE(generated, expert) + small jerk penalty (same for both arms)."""
        B = expert_chunk.shape[0]
        if noise is None:
            noise = torch.randn_like(expert_chunk)
        gen = self.forward(cond, noise)
        mse = F.mse_loss(gen, expert_chunk)
        jerk = (gen[:, 2:] - 2 * gen[:, 1:-1] + gen[:, :-2]).pow(2).mean()  # 2nd diff = accel; penalize roughness
        return mse + smooth * jerk, {"mse": float(mse), "jerk": float(jerk)}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, cond_dim, H, A = 8, 39, 16, 4
    cond = torch.randn(B, cond_dim); expert = torch.randn(B, H, A) * 0.3
    print("=== self-test: both arms run, finite, stable; param counts ===")
    for mode in ("vanilla", "impedance"):
        pol = CompactFlowPolicy(cond_dim, A, H, mode=mode)
        noise = torch.randn(B, H, A)
        out = pol(cond, noise)
        L, info = pol.loss(cond, expert, noise)
        finite = bool(torch.isfinite(out).all())
        print(f"  {mode:9s}: out{tuple(out.shape)} finite={finite} max|a|={float(out.abs().max()):6.2f} "
              f"loss={float(L):.3f} params={n_params(pol):,}")
        assert finite and out.shape == (B, H, A)
        L.backward()  # gradients flow through the rollout
    print("  PASS: both arms produce finite, stable chunks and are differentiable through the ODE.")
