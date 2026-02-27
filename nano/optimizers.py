"""Muon / Shampoo / Adam optimizers for KataGo nano training."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Newton-Schulz coefficients for matrix orthogonalization (polar_express)
# ---------------------------------------------------------------------------
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Newton-Schulz coefficients for inverse 4th root (Shampoo preconditioner)
_NS_COEFFS_R4_SCALED = (
    (3.7745392156862745, -9.830711636812923, 7.211935063687831),
    (1.7744313725490195, -0.5323686439402083, 0.05420935725061334),
    (1.4744509803921568, -0.5384714581368423, 0.10138210476839715),
    (1.3786764705882353, -0.5094735805293277, 0.13074301029260285),
)


@torch.compile
def polar_express(G):
    """Newton-Schulz iteration for matrix orthogonalization, 5 steps."""
    assert G.ndim == 2
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)

    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


class MuonOptimizer:
    """Muon optimizer: momentum + Newton-Schulz orthogonalization."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, scale_mode="moonlight", device="cuda"):
        self.named_params = named_params
        self.lr_multiplier = lr_multiplier
        self.momentum = momentum
        self.wd = wd
        self.scale_mode = scale_mode
        self._device = device
        self.last_update_rms = 0.0
        self.states = {name: self._init_state(p) for name, p in named_params.items()}

    @staticmethod
    def _init_state(p):
        return {"momentum": torch.zeros_like(p)}

    def step(self, base_lr):
        muon_lr = base_lr * self.lr_multiplier
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                assert p.grad.ndim in (2, 4), f"Muon only supports 2D/4D params, got ndim={p.grad.ndim}"
                state = self.states[name]
                grad = p.grad
                original_shape = grad.shape

                state["momentum"].mul_(self.momentum).add_(grad)
                update = state["momentum"]
                if update.ndim == 4:
                    update = update.view(update.size(0), -1)

                update = polar_express(update)
                if self.scale_mode == "moonlight":
                    update = update * max(update.size()) ** 0.5
                else:
                    update = update * max(1, update.size(0) / update.size(1)) ** 0.5
                update = update.view(original_shape)

                rms_sum += update.norm() * self.lr_multiplier / update.numel() ** 0.5
                rms_cnt += 1

                p.mul_(1 - base_lr * self.wd)
                p.add_(update.to(p.dtype), alpha=-muon_lr)

        self.last_update_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        return {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}

    def load_state_dict(self, saved, device):
        for name, tensors in saved.items():
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


@torch.compile
def inv_quarter_sandwich(L, M, R):
    """Newton-Schulz iteration for L^{-1/4} @ M @ R^{-1/4}, 4 steps, fp32."""
    assert L.ndim == 2 and M.ndim == 2 and R.ndim == 2
    eps = 1e-4
    M = M.float()

    m = L.size(-1)
    n = R.size(-1)
    I_L = torch.eye(m, device=L.device)
    I_R = torch.eye(n, device=L.device)

    tL = torch.sqrt((L * L.mT).sum())
    tR = torch.sqrt((R * R.mT).sum())
    L = L / tL + eps * I_L
    R = R / tR + eps * I_R

    for a, b, c in _NS_COEFFS_R4_SCALED:
        L2 = L @ L
        WL = a * I_L + b * L + c * L2

        R2 = R @ R
        WR = a * I_R + b * R + c * R2

        M = WL @ M @ WR

        WL4 = (WL @ WL) @ (WL @ WL)
        WR4 = (WR @ WR) @ (WR @ WR)
        L = L @ WL4
        R = R @ WR4

    M = M * (tL ** (-0.25)) * (tR ** (-0.25))
    return M


class ShampooOptimizer:
    """Shampoo optimizer: L/R preconditioner EMA + matrix inverse root."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, beta2=0.999, device="cuda"):
        self.named_params = named_params
        self.lr_multiplier = lr_multiplier
        self.momentum = momentum
        self.wd = wd
        self.beta2 = beta2
        self.step_count = 0
        self._device = device
        self.last_precond_rms = 0.0
        self.states = {name: self._init_state(p, device) for name, p in named_params.items()}

    @staticmethod
    def _init_state(p, device):
        if p.ndim >= 2:
            m, n = p.shape[0], p.shape[1:].numel()
        else:
            m, n = p.shape[0], 1
        return {
            "momentum": torch.zeros_like(p),
            "L": torch.zeros(m, m, dtype=torch.float32, device=device),
            "R": torch.zeros(n, n, dtype=torch.float32, device=device),
        }

    def step(self, base_lr):
        self.step_count += 1
        shampoo_lr = base_lr * self.lr_multiplier
        bias_corr1 = 1 - self.momentum ** self.step_count
        bias_corr2 = 1 - self.beta2 ** self.step_count
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                assert p.grad.ndim in (2, 4), f"Shampoo only supports 2D/4D params, got ndim={p.grad.ndim}"
                state = self.states[name]
                grad = p.grad
                original_shape = grad.shape
                if grad.ndim == 4:
                    grad_2d = grad.view(grad.size(0), -1)
                else:
                    grad_2d = grad

                state["momentum"].lerp_(grad, 1 - self.momentum)
                momentum_2d = state["momentum"]
                if momentum_2d.ndim == 4:
                    momentum_2d = momentum_2d.view(momentum_2d.size(0), -1)
                momentum_2d_hat = momentum_2d / bias_corr1

                state["L"].lerp_(grad_2d @ grad_2d.mT, 1 - self.beta2)
                state["R"].lerp_(grad_2d.mT @ grad_2d, 1 - self.beta2)

                precond = inv_quarter_sandwich(
                    state["L"] / bias_corr2, momentum_2d_hat, state["R"] / bias_corr2,
                )
                precond = precond * (precond.size(0) * precond.size(1)) ** 0.25

                rms_sum += precond.norm() * self.lr_multiplier / precond.numel() ** 0.5
                rms_cnt += 1
                precond = precond.view(original_shape)

                p.mul_(1 - base_lr * self.wd)
                p.add_(precond.to(p.dtype), alpha=-shampoo_lr)

        self.last_precond_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        result = {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}
        result["__step_count__"] = self.step_count
        return result

    def load_state_dict(self, saved, device):
        self.step_count = saved.get("__step_count__", 0)
        for name, tensors in saved.items():
            if name == "__step_count__":
                continue
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


class AdamOptimizer:
    """AdamW-style optimizer with decoupled weight decay, tracking update RMS."""

    def __init__(self, named_params, wd, beta1=0.9, beta2=0.95, eps=1e-8, device="cuda"):
        self.named_params = named_params
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self._device = device
        self.last_update_rms = 0.0
        self.states = {name: self._init_state(p) for name, p in named_params.items()}

    @staticmethod
    def _init_state(p):
        return {
            "m": torch.zeros_like(p),
            "v": torch.zeros_like(p),
        }

    def step(self, base_lr):
        self.step_count += 1
        bias_corr1 = 1 - self.beta1 ** self.step_count
        bias_corr2 = 1 - self.beta2 ** self.step_count
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                state = self.states[name]
                grad = p.grad

                state["m"].lerp_(grad, 1 - self.beta1)
                state["v"].lerp_(grad * grad, 1 - self.beta2)

                m_hat = state["m"] / bias_corr1
                v_hat = state["v"] / bias_corr2

                update = m_hat / (v_hat.sqrt() + self.eps)

                rms_sum += update.norm() / update.numel() ** 0.5
                rms_cnt += 1

                p.mul_(1 - base_lr * self.wd)
                p.add_(update, alpha=-base_lr)

        self.last_update_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        result = {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}
        result["__step_count__"] = self.step_count
        return result

    def load_state_dict(self, saved, device):
        self.step_count = saved.get("__step_count__", 0)
        for name, tensors in saved.items():
            if name == "__step_count__":
                continue
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))
