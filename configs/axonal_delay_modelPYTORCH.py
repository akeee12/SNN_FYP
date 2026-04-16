

# ─────────────────────────────────────────────────────────────────────────────
# FIXED: DIFFERENTIABLE DELAY GRADIENT (NO DOUBLE SHIFT)
# ─────────────────────────────────────────────────────────────────────────────

class DelayGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikes, delay):
        # Forward is identity — delay already applied via interpolation
        ctx.save_for_backward(spikes, delay)
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        spikes, delay = ctx.saved_tensors

        # Gradient w.r.t spikes (identity)
        grad_spikes = grad_output.clone()

        # Finite difference gradient w.r.t delay
        s_pad = F.pad(spikes, (1, 1))
        ds_dt = (s_pad[:, :, 2:] - s_pad[:, :, :-2]) / 2.0

        grad_delay = (grad_output * ds_dt).sum(dim=(0, 2))

        return grad_spikes, grad_delay


# ─────────────────────────────────────────────────────────────────────────────
# AXONAL DELAY (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

class AxonalDelay(nn.Module):
    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.n_neurons = n_neurons
        self.max_delay = max_delay

        self.delay = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 8)
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        d = self.delay.clamp(0, self.max_delay)

        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor()).view(1, -1, 1)

        B, N, T = spikes.shape

        out_floor = torch.zeros_like(spikes)
        out_ceil  = torch.zeros_like(spikes)

        for i in range(N):
            sf = d_floor[i].item()
            sc = d_ceil[i].item()

            if sf > 0:
                out_floor[:, i, sf:] = spikes[:, i, :T - sf]
            else:
                out_floor[:, i, :] = spikes[:, i, :]

            if sc > 0:
                out_ceil[:, i, sc:] = spikes[:, i, :T - sc]
            else:
                out_ceil[:, i, :] = spikes[:, i, :]

        out = (1 - alpha) * out_floor + alpha * out_ceil

        # IMPORTANT: Only apply gradient logic — no extra shift
        return DelayGradient.apply(out, d)

    def clamp_delays(self, max_val: float):
        with torch.no_grad():
            self.delay.data.clamp_(0, max_val)

    def get_stats(self):
        d = self.delay.detach().cpu()
        return {
            "max": d.max().item(),
            "mean": d.mean().item(),
            "p95": torch.quantile(d, 0.95).item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PSP + REFRACTORY (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

class PSPKernel(nn.Module):
    def __init__(self, tau_s: float = 1.0, kernel_size: int = 32):
        super().__init__()
        t = torch.arange(kernel_size, dtype=torch.float32)
        kernel = torch.exp(-t / tau_s)
        kernel = kernel / (kernel.sum() + 1e-8)
        self.register_buffer("kernel", kernel.flip(0).view(1, 1, -1))
        self.kernel_size = kernel_size

    def forward(self, x):
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.conv1d(x, self.kernel)
        return x.reshape(B, N, T)


class RefractoryKernel(nn.Module):
    def __init__(self, tau_r=1.0, theta_u=10.0, alpha_r=2.0, kernel_size=32):
        super().__init__()
        t = torch.arange(kernel_size, dtype=torch.float32)
        kernel = -theta_u * alpha_r * torch.exp(-t / tau_r)
        self.register_buffer("kernel", kernel.flip(0).view(1, 1, -1))
        self.kernel_size = kernel_size

    def forward(self, spikes):
        B, N, T = spikes.shape
        s = spikes.reshape(B * N, 1, T)
        s = F.pad(s, (self.kernel_size - 1, 0))
        s = F.conv1d(s, self.kernel)
        return s.reshape(B, N, T)


# ─────────────────────────────────────────────────────────────────────────────
# SRM LAYER (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

class SLAYERSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, theta, tau_rho, scale_rho):
        ctx.save_for_backward(u)
        ctx.theta = theta
        ctx.tau_rho = tau_rho
        ctx.scale_rho = scale_rho
        return (u >= theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        theta = ctx.theta
        tau_rho = ctx.tau_rho
        scale_rho = ctx.scale_rho
        surrogate = (scale_rho / tau_rho) * \
            torch.clamp(1 - torch.abs(u - theta) / tau_rho, min=0)
        return grad_output * surrogate, None, None, None


class SRMLayer(nn.Module):
    def __init__(self, tau_s=1.0, tau_r=1.0, theta_u=10.0,
                 alpha_r=2.0, tau_rho=1.0, scale_rho=1.0, kernel_size=32):
        super().__init__()
        self.theta_u = theta_u
        self.tau_rho = tau_rho
        self.scale_rho = scale_rho

        self.psp = PSPKernel(tau_s, kernel_size)
        self.refractory = RefractoryKernel(tau_r, theta_u, alpha_r, kernel_size)

    def forward(self, x):
        u = self.psp(x)
        B, N, T = u.shape
        spikes = torch.zeros_like(u)

        for t in range(T):
            if t > 0:
                ref = self.refractory(
                    F.pad(spikes[:, :, :t], (0, T - t))
                )[:, :, t]
            else:
                ref = torch.zeros(B, N, device=u.device)

            u_t = u[:, :, t] + ref
            s_t = SLAYERSurrogate.apply(
                u_t, self.theta_u, self.tau_rho, self.scale_rho
            )
            spikes[:, :, t] = s_t

        return spikes


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAxonalDelayNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        srm_kwargs = dict(
            tau_s=cfg.tau_s,
            tau_r=cfg.tau_r,
            theta_u=cfg.theta_u,
            alpha_r=cfg.alpha_r,
            tau_rho=cfg.tau_rho,
            scale_rho=cfg.scale_rho,
            kernel_size=32,
        )

        self.srm1 = SRMLayer(**srm_kwargs)
        self.srm2 = SRMLayer(**srm_kwargs)
        self.srm3 = SRMLayer(**srm_kwargs)

        self.fc1 = nn.utils.weight_norm(nn.Linear(cfg.n_inputs, cfg.n_hidden, bias=False))
        self.fc2 = nn.utils.weight_norm(nn.Linear(cfg.n_hidden, cfg.n_hidden, bias=False))
        self.fc3 = nn.utils.weight_norm(nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=False))

        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.srm1(x)
        x = self.delay1(x)

        x = self.fc2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.srm2(x)
        x = self.delay2(x)

        x = self.fc3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.srm3(x)

        return x.sum(dim=-1)

    def clamp_delays(self, thea1, thea2):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self):
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }