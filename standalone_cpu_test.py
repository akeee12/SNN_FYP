import sys
import time
import tempfile
import os
import traceback
import numpy as np
import torch
import torch.nn as nn

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg):  print(f"  {YELLOW}-->{RESET}  {msg}")
def header(msg):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")

PASS       = 0
FAIL_COUNT = 0

def run_test(name, fn):
    global PASS, FAIL_COUNT
    try:
        fn()
        ok(name)
        PASS += 1
    except Exception as e:
        fail(name)
        print(f"       {RED}{type(e).__name__}: {e}{RESET}")
        traceback.print_exc()
        FAIL_COUNT += 1

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GaussianSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, threshold, sigma, scale):
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        ctx.sigma     = sigma
        ctx.scale     = scale
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        u,        = ctx.saved_tensors
        surrogate = ctx.scale * torch.exp(
            -(u - ctx.threshold) ** 2 / (2 * ctx.sigma ** 2 + 1e-8)
        )
        return grad_output * surrogate, None, None, None

def gaussian_spike(u, threshold=1.0, sigma=0.5, scale=1.0):
    return GaussianSurrogate.apply(u, threshold, sigma, scale)


class SynapticDelay(nn.Module):
    def __init__(self, n_pre, n_post, max_delay=20):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_post, n_pre).uniform_(0, max_delay / 4)
        )

    def forward(self, x):
        B, T, N = x.shape
        d       = self.delay.mean(dim=0).clamp(0, self.max_delay)
        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor()).view(1, 1, N)
        pad     = torch.zeros(B, self.max_delay, N,
                              device=x.device, dtype=x.dtype)
        padded  = torch.cat([pad, x], dim=1)
        t_base  = torch.arange(T, device=x.device).view(1, T, 1)
        start_f = (self.max_delay - d_floor).clamp(0).view(1, 1, N)
        start_c = (self.max_delay - d_ceil ).clamp(0).view(1, 1, N)
        idx_f   = (t_base + start_f).expand(B, T, N)
        idx_c   = (t_base + start_c).expand(B, T, N)
        return (1 - alpha) * torch.gather(padded, 1, idx_f) \
                   + alpha * torch.gather(padded, 1, idx_c)

    def get_stats(self):
        d = self.delay.detach().cpu()
        return {"max" : d.max().item(),
                "mean": d.mean().item(),
                "p95" : torch.quantile(d.flatten(), 0.95).item()}


class AxonalDelay(nn.Module):
    def __init__(self, n_neurons, max_delay=20):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 4)
        )

    def forward(self, spikes):
        B, T, N = spikes.shape
        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor()).view(1, 1, N)
        pad     = torch.zeros(B, self.max_delay, N,
                              device=spikes.device, dtype=spikes.dtype)
        padded  = torch.cat([pad, spikes], dim=1)
        t_base  = torch.arange(T, device=spikes.device).view(1, T, 1)
        start_f = (self.max_delay - d_floor).clamp(0).view(1, 1, N)
        start_c = (self.max_delay - d_ceil ).clamp(0).view(1, 1, N)
        idx_f   = (t_base + start_f).expand(B, T, N)
        idx_c   = (t_base + start_c).expand(B, T, N)
        return (1 - alpha) * torch.gather(padded, 1, idx_f) \
                   + alpha * torch.gather(padded, 1, idx_c)

    def get_stats(self):
        d = self.delay.detach().cpu()
        return {"max" : d.max().item(),
                "mean": d.mean().item(),
                "p95" : torch.quantile(d, 0.95).item()}


class LIFNeuron(nn.Module):
    def __init__(self, tau_mem=20.0, tau_syn=5.0,
                 v_threshold=1.0, v_reset=0.0,
                 surrogate_sigma=0.5, surrogate_scale=1.0, dt=1.0):
        super().__init__()
        self.v_threshold     = v_threshold
        self.v_reset         = v_reset
        self.surrogate_sigma = surrogate_sigma
        self.surrogate_scale = surrogate_scale
        self.decay_mem       = np.exp(-dt / tau_mem)
        self.decay_syn       = np.exp(-dt / tau_syn)

    def forward(self, x):
        B, T, N = x.shape
        u      = torch.zeros(B, N, device=x.device)
        i_syn  = torch.zeros(B, N, device=x.device)
        spikes = torch.zeros(B, T, N, device=x.device)
        for t in range(T):
            i_syn           = i_syn * self.decay_syn + x[:, t, :]
            u               = u     * self.decay_mem + i_syn
            s               = gaussian_spike(u, self.v_threshold,
                                             self.surrogate_sigma,
                                             self.surrogate_scale)
            spikes[:, t, :] = s
            u               = u * (1 - s.detach()) + self.v_reset * s.detach()
        return spikes


class ThreeFactorDelayNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        lif_kw = dict(tau_mem=cfg.tau_mem, tau_syn=cfg.tau_syn,
                      v_threshold=cfg.v_threshold, v_reset=cfg.v_reset,
                      surrogate_sigma=cfg.surrogate_sigma,
                      surrogate_scale=cfg.surrogate_scale)
        self.fc1    = nn.Linear(cfg.n_inputs,  cfg.n_hidden, bias=False)
        self.sdel1  = SynapticDelay(cfg.n_inputs,  cfg.n_hidden,
                                    max_delay=cfg.max_synaptic_delay)
        self.lif1   = LIFNeuron(**lif_kw)
        self.adel1  = AxonalDelay(cfg.n_hidden, max_delay=cfg.max_axonal_delay)
        self.fc2    = nn.Linear(cfg.n_hidden,  cfg.n_hidden, bias=False)
        self.sdel2  = SynapticDelay(cfg.n_hidden,  cfg.n_hidden,
                                    max_delay=cfg.max_synaptic_delay)
        self.lif2   = LIFNeuron(**lif_kw)
        self.adel2  = AxonalDelay(cfg.n_hidden, max_delay=cfg.max_axonal_delay)
        self.fc_out = nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=False)
        self.drop   = nn.Dropout(cfg.dropout_p) if cfg.dropout_p > 0 \
                      else nn.Identity()

    def forward(self, x):
        x = self.adel1(self.lif1(self.sdel1(self.fc1(x))))
        x = self.drop(x)
        x = self.adel2(self.lif2(self.sdel2(self.fc2(x))))
        x = self.drop(x)
        return self.fc_out(x).sum(dim=1)

    def get_delay_stats(self):
        return {"synaptic_delay1": self.sdel1.get_stats(),
                "axonal_delay1"  : self.adel1.get_stats(),
                "synaptic_delay2": self.sdel2.get_stats(),
                "axonal_delay2"  : self.adel2.get_stats()}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TinyCfg:
    model_name         = "ThreeFactorDelay"
    n_inputs           = 700
    n_hidden           = 32
    n_outputs          = 20
    time_steps         = 50
    tau_mem            = 20.0
    tau_syn            = 5.0
    v_threshold        = 1.0
    v_reset            = 0.0
    max_synaptic_delay = 10
    max_axonal_delay   = 10
    learning_rate_d    = 0.01
    surrogate_sigma    = 0.5
    surrogate_scale    = 1.0
    lr                 = 0.001
    dropout_p          = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_gaussian_surrogate():
    u = torch.tensor([-1.0, 0.5, 1.0, 1.5, 2.0], requires_grad=True)
    s = gaussian_spike(u, threshold=1.0, sigma=0.5, scale=1.0)
    assert torch.allclose(s, torch.tensor([0., 0., 1., 1., 1.])), \
        f"Heaviside wrong: {s}"
    s.sum().backward()
    assert u.grad is not None, "No gradient"
    assert u.grad.detach()[2] > u.grad.detach()[0], \
        "Gradient not peaked at threshold"

def test_synaptic_delay_shape():
    sd = SynapticDelay(10, 8, max_delay=5)
    y  = sd(torch.randn(2, 20, 10))
    assert y.shape == (2, 20, 10), f"Shape wrong: {y.shape}"

def test_synaptic_delay_shifts():
    sd = SynapticDelay(4, 4, max_delay=10)
    with torch.no_grad():
        sd.delay.fill_(5.0)
    x = torch.zeros(1, 30, 4)
    x[:, 0, :] = 1.0
    y = sd(x)
    assert y[:, 5, :].abs().sum() > y[:, :4, :].abs().sum(), \
        "SynapticDelay not shifting signal"

def test_axonal_delay_shape():
    ad = AxonalDelay(16, max_delay=5)
    y  = ad(torch.zeros(2, 20, 16))
    assert y.shape == (2, 20, 16), f"Shape wrong: {y.shape}"

def test_axonal_delay_shifts():
    ad = AxonalDelay(4, max_delay=10)
    with torch.no_grad():
        ad.delay.fill_(5.0)
    spikes = torch.zeros(1, 30, 4)
    spikes[:, 0, :] = 1.0
    y = ad(spikes)
    assert y[:, 5, :].abs().sum() > y[:, :4, :].abs().sum(), \
        "AxonalDelay not shifting spikes"

def test_lif_neuron():
    lif    = LIFNeuron()
    spikes = lif(torch.randn(2, 50, 32) * 0.5)
    assert spikes.shape == (2, 50, 32), f"Shape wrong: {spikes.shape}"
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0}), \
        "Non-binary spikes"
    assert spikes.sum() > 0, "Zero spikes produced"

def test_model_instantiation():
    model = ThreeFactorDelayNet(TinyCfg())
    for attr in ["fc1","sdel1","lif1","adel1",
                 "fc2","sdel2","lif2","adel2","fc_out"]:
        assert hasattr(model, attr), f"Missing: {attr}"

def test_parameter_count():
    cfg      = TinyCfg()
    model    = ThreeFactorDelayNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    expected = (cfg.n_inputs * cfg.n_hidden +
                cfg.n_hidden * cfg.n_inputs +
                cfg.n_hidden               +
                cfg.n_hidden * cfg.n_hidden +
                cfg.n_hidden * cfg.n_hidden +
                cfg.n_hidden               +
                cfg.n_hidden * cfg.n_outputs)
    info(f"Params: {n_params:,}  expected: {expected:,}")
    assert n_params == expected, \
        f"Mismatch: got {n_params}, expected {expected}"
    delay_p = sum(p.numel() for n, p in model.named_parameters()
                  if "delay" in n)
    info(f"Delay params: {delay_p:,} ({100*delay_p/n_params:.1f}% of total)")

def test_forward_shape():
    cfg   = TinyCfg()
    model = ThreeFactorDelayNet(cfg)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(4, cfg.time_steps, cfg.n_inputs))
    assert out.shape == (4, cfg.n_outputs), f"Shape wrong: {out.shape}"

def test_forward_nonzero():
    cfg   = TinyCfg()
    model = ThreeFactorDelayNet(cfg)
    model.eval()
    x = (torch.rand(4, cfg.time_steps, cfg.n_inputs) < 0.05).float()
    with torch.no_grad():
        out = model(x)
    assert out.abs().sum() > 0, "All-zero output with non-zero input"

def test_loss():
    cfg       = TinyCfg()
    model     = ThreeFactorDelayNet(cfg)
    criterion = nn.CrossEntropyLoss()
    x         = (torch.rand(4, cfg.time_steps, cfg.n_inputs) < 0.05).float()
    labels    = torch.randint(0, cfg.n_outputs, (4,))
    loss      = criterion(model(x), labels)
    assert loss.item() > 0,       "Loss is zero"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

def test_backward():
    cfg       = TinyCfg()
    model     = ThreeFactorDelayNet(cfg)
    criterion = nn.CrossEntropyLoss()
    x         = (torch.rand(4, cfg.time_steps, cfg.n_inputs) < 0.05).float()
    labels    = torch.randint(0, cfg.n_outputs, (4,))
    criterion(model(x), labels).backward()
    no_grad  = [n for n, p in model.named_parameters() if p.grad is None]
    nan_grad = [n for n, p in model.named_parameters()
                if p.grad is not None and torch.isnan(p.grad).any()]
    if no_grad:  raise AssertionError(f"No gradient: {no_grad}")
    if nan_grad: raise AssertionError(f"NaN gradient: {nan_grad}")

def test_delay_clamping():
    cfg   = TinyCfg()
    model = ThreeFactorDelayNet(cfg)
    opt   = torch.optim.Adam(model.parameters(), lr=0.1)
    with torch.no_grad():
        model.sdel1.delay.fill_(999.0)
        model.adel1.delay.fill_(-99.0)
    x      = (torch.rand(2, cfg.time_steps, cfg.n_inputs) < 0.05).float()
    labels = torch.randint(0, cfg.n_outputs, (2,))
    nn.CrossEntropyLoss()(model(x), labels).backward()
    opt.step()
    with torch.no_grad():
        model.sdel1.delay.clamp_(0, cfg.max_synaptic_delay)
        model.sdel2.delay.clamp_(0, cfg.max_synaptic_delay)
        model.adel1.delay.clamp_(0, cfg.max_axonal_delay)
        model.adel2.delay.clamp_(0, cfg.max_axonal_delay)
    assert model.sdel1.delay.min() >= 0
    assert model.sdel1.delay.max() <= cfg.max_synaptic_delay
    assert model.adel1.delay.min() >= 0
    assert model.adel1.delay.max() <= cfg.max_axonal_delay

def test_delay_stats():
    stats = ThreeFactorDelayNet(TinyCfg()).get_delay_stats()
    for key in ["synaptic_delay1","axonal_delay1",
                "synaptic_delay2","axonal_delay2"]:
        assert key in stats, f"Missing: {key}"
        for sub in ["max","mean","p95"]:
            assert sub in stats[key], f"Missing {sub} in {key}"
    info("Delay stats at init:")
    for layer, s in stats.items():
        info(f"  {layer:20s}: mean={s['mean']:.3f} "
             f"max={s['max']:.3f} p95={s['p95']:.3f}")

def test_separate_lr():
    cfg      = TinyCfg()
    model    = ThreeFactorDelayNet(cfg)
    delay_p  = [p for n, p in model.named_parameters() if "delay" in n]
    weight_p = [p for n, p in model.named_parameters() if "delay" not in n]
    opt = torch.optim.Adam([{"params": weight_p, "lr": cfg.lr},
                             {"params": delay_p,  "lr": cfg.learning_rate_d}])
    assert opt.param_groups[0]["lr"] == cfg.lr
    assert opt.param_groups[1]["lr"] == cfg.learning_rate_d
    info(f"Weight LR: {opt.param_groups[0]['lr']}  "
         f"Delay LR: {opt.param_groups[1]['lr']}")

def test_mini_training_loop():
    cfg       = TinyCfg()
    model     = ThreeFactorDelayNet(cfg)
    criterion = nn.CrossEntropyLoss()
    delay_p   = [p for n, p in model.named_parameters() if "delay" in n]
    weight_p  = [p for n, p in model.named_parameters() if "delay" not in n]
    opt       = torch.optim.Adam([{"params": weight_p, "lr": cfg.lr},
                                   {"params": delay_p,  "lr": cfg.learning_rate_d}])
    for epoch in range(3):
        model.train()
        x      = (torch.rand(4, cfg.time_steps, cfg.n_inputs) < 0.05).float()
        labels = torch.randint(0, cfg.n_outputs, (4,))
        opt.zero_grad()
        loss = criterion(model(x), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            model.sdel1.delay.clamp_(0, cfg.max_synaptic_delay)
            model.sdel2.delay.clamp_(0, cfg.max_synaptic_delay)
            model.adel1.delay.clamp_(0, cfg.max_axonal_delay)
            model.adel2.delay.clamp_(0, cfg.max_axonal_delay)
        info(f"  Epoch {epoch+1}: loss={loss.item():.5f}")
        assert np.isfinite(loss.item()), f"Non-finite loss at epoch {epoch+1}"

def test_checkpoint_roundtrip():
    cfg   = TinyCfg()
    model = ThreeFactorDelayNet(cfg)
    fc1_w = model.fc1.weight.data.clone()
    sd1_d = model.sdel1.delay.data.clone()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        torch.save({"epoch": 1, "model_state": model.state_dict()}, path)
        model2 = ThreeFactorDelayNet(cfg)
        model2.load_state_dict(
            torch.load(path, map_location="cpu")["model_state"]
        )
    assert torch.allclose(fc1_w, model2.fc1.weight.data), "FC1 changed!"
    assert torch.allclose(sd1_d, model2.sdel1.delay.data), "sdel1 changed!"

def test_vectorised_speed():
    ad     = AxonalDelay(128, max_delay=20)
    spikes = torch.randn(8, 400, 128)
    _      = ad(spikes)
    t0     = time.time()
    for _  in range(20):
        _  = ad(spikes)
    ms = (time.time() - t0) / 20 * 1000
    info(f"AxonalDelay (B=8, T=400, N=128): {ms:.2f} ms per call")
    assert ms < 2000, f"Too slow: {ms:.1f} ms"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Three-Factor Delay SNN — CPU Test")
    print(f"{'='*60}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()} (not used here)")

    tests = [
        ("Gaussian surrogate fwd/bwd",          test_gaussian_surrogate),
        ("SynapticDelay output shape",           test_synaptic_delay_shape),
        ("SynapticDelay actually shifts signal", test_synaptic_delay_shifts),
        ("AxonalDelay output shape",             test_axonal_delay_shape),
        ("AxonalDelay actually shifts spikes",   test_axonal_delay_shifts),
        ("LIF neuron forward",                   test_lif_neuron),
        ("Model instantiation",                  test_model_instantiation),
        ("Parameter count",                      test_parameter_count),
        ("Forward shape zero input",             test_forward_shape),
        ("Forward nonzero output",               test_forward_nonzero),
        ("Loss computation",                     test_loss),
        ("Backward gradient flow",               test_backward),
        ("Delay clamping",                       test_delay_clamping),
        ("Delay stats",                          test_delay_stats),
        ("Separate delay/weight LR",             test_separate_lr),
        ("Mini training loop 3 epochs",          test_mini_training_loop),
        ("Checkpoint round-trip",                test_checkpoint_roundtrip),
        ("Vectorised delay speed",               test_vectorised_speed),
    ]

    for name, fn in tests:
        header(name)
        run_test(name, fn)

    total = PASS + FAIL_COUNT
    print(f"\n{'='*60}")
    print(f"  Results : {PASS}/{total} passed")
    if FAIL_COUNT == 0:
        print(f"  {GREEN}All tests passed — safe to submit HPC job!{RESET}")
    else:
        print(f"  {RED}{FAIL_COUNT} test(s) FAILED — fix before submitting!{RESET}")
    print(f"{'='*60}\n")
    return FAIL_COUNT

if __name__ == "__main__":
    exit(main())
