"""
hebbattn.py — Supervised/unsupervised Hebbian learning from noisy examples
in pairwise Hopfield vs exponential (softmax / attention) Hopfield models.

Setting (Alemanno-Aquaro-Kanter-Barra-Agliari, EPL 2023):
  K archetypes xi^mu in {±1}^N (Rademacher), M examples per archetype
  eta^{mu a} = xi^mu * chi^{mu a},  P(chi=+1)=(1+r)/2.
  Control parameters: alpha=K/N, rho=(1-r^2)/(M r^2).

Models (all zero-temperature parallel dynamics sigma <- sign(h(sigma))):
  pairwise_sup   : h_i = sum_mu etabar^mu_i (etabar^mu . sigma)          [EPL 2023, eq. 4]
  pairwise_unsup : h_i = sum_{mu,a} eta^{mu a}_i (eta^{mu a} . sigma)     [EPL 2023, eq. 5]
  attn_sup       : h   = sum_mu  softmax_mu( lam * N * m_mu(sigma) ) etabar^mu
  attn_unsup     : h   = sum_{mu,a} softmax_{mu a}( lam * N * m_{mu a}(sigma) ) eta^{mu a}
with m_x(sigma) = x . sigma / N the overlap with the key x.
attn_unsup is exactly one-head self-attention with keys=values=training examples and
query=current state (Ramsauer et al. 2021 / Lucibello-Mezard exponential model); lam is
the inverse temperature in units where the logit is lam*N*m, so lam -> 0 recovers the
pairwise Hebbian (unsupervised) rule up to a constant mean-key term, lam -> inf is
nearest-training-example lookup.
"""
import numpy as np
from scipy.special import erf, logsumexp


# ----------------------------------------------------------------------------- data
def make_dataset(N, K, M, r, rng):
    xi = rng.choice([-1, 1], size=(K, N)).astype(np.int8)
    chi = np.where(rng.random((K, M, N)) < (1 + r) / 2, 1, -1).astype(np.int8)
    eta = xi[:, None, :] * chi  # (K, M, N)
    return xi, eta


def noisy_copy(xi_mu, r, rng):
    chi = np.where(rng.random(xi_mu.shape) < (1 + r) / 2, 1, -1).astype(np.int8)
    return xi_mu * chi


# --------------------------------------------------------------------------- models
class HebbianMemory:
    """Container for keys (and labels). Provides the four update maps."""

    def __init__(self, xi, eta, model, lam=None):
        self.model = model
        self.lam = lam
        K, M, N = eta.shape
        self.K, self.M, self.N = K, M, N
        self.xi = xi.astype(np.float32)
        self.eta_all = eta.reshape(K * M, N).astype(np.float32)
        self.labels_all = np.repeat(np.arange(K), M)
        self.etabar = eta.astype(np.float32).mean(axis=1)  # (K, N) class means
        if model in ("pairwise_sup", "attn_sup"):
            self.keys, self.labels = self.etabar, np.arange(K)
        elif model in ("pairwise_unsup", "attn_unsup"):
            self.keys, self.labels = self.eta_all, self.labels_all
        else:
            raise ValueError(model)
        self.diag = (self.keys ** 2).sum(axis=0)  # J_ii, removed in the pairwise models

    # ---- one update step; also return softmax weights (attention) for diagnostics
    def field(self, sigma):
        over = self.keys @ sigma / self.N  # overlaps m_x(sigma), shape (P,)
        if self.model.startswith("pairwise"):
            w = over  # linear Hebbian weights
            h = w @ self.keys - self.diag * sigma / self.N  # no self-interaction J_ii
            return h, None
        logits = self.lam * self.N * over
        logw = logits - logsumexp(logits)
        w = np.exp(logw)
        h = w @ self.keys
        return h, w

    def step(self, sigma):
        h, w = self.field(sigma)
        new = np.sign(h)
        new[new == 0] = sigma[new == 0]
        return new.astype(np.float32), w

    def run(self, sigma0, T=10):
        """Returns (sigma after one step, sigma at convergence/T, attention weights at step 1)."""
        sigma = sigma0.astype(np.float32).copy()
        w_first, s1 = None, None
        for t in range(T):
            new, w = self.step(sigma)
            if t == 0:
                w_first, s1 = w, new.copy()
            if np.array_equal(new, sigma):
                break
            sigma = new
        return s1, sigma, w_first


# ----------------------------------------------------------------------- diagnostics
def diagnostics(mem, sigma, mu, w, r):
    """Overlap with archetype, with closest training example, predicted class,
    effective number of pooled same-class examples, rho_eff."""
    N = mem.N
    m = float(mem.xi[mu] @ sigma / N)
    same = mem.eta_all[mem.labels_all == mu]
    n_max_same = float((same @ sigma / N).max())
    n_max_all = float((mem.eta_all @ sigma / N).max())
    out = dict(m=m, n_max_same=n_max_same, n_max_all=n_max_all)
    if w is not None:
        cls_w = np.bincount(mem.labels, weights=w, minlength=mem.K)
        out["pred"] = int(np.argmax(cls_w))
        out["w_true_class"] = float(cls_w[mu])
        if mem.model == "attn_unsup":
            ww = w[mem.labels == mu]
            W = ww.sum()
            out["M_eff"] = float(W ** 2 / np.sum(ww ** 2))  # participation ratio
            # effective dataset entropy of the attention readout:
            # signal r*W, noise var (1-r^2) sum w_a^2 (same class) + sum v_b^2 (other classes)
            vv = w[mem.labels != mu]
            lam = mem.lam
            # the self-overlap term in the logit gives a self-coupling ~ lam*sigma_i (first order)
            # other-class keys of the same class share an archetype -> their weighted sum has
            # variance r^2 V_nu^2 + (1-r^2) sum_b v_b^2 per class nu (not simply sum v_b^2)
            V = np.bincount(mem.labels, weights=w, minlength=mem.K)
            Vsq = np.bincount(mem.labels, weights=w ** 2, minlength=mem.K)
            mask = np.arange(mem.K) != mu
            noise_var = float(np.sum(r ** 2 * V[mask] ** 2 + (1 - r ** 2) * Vsq[mask]))
            mean = r * (W + lam)
            var = (1 - r ** 2) * (np.sum(ww ** 2) + lam ** 2) + noise_var
            out["rho_eff"] = float(var / mean ** 2)
            out["rho_eff0"] = float(((1 - r ** 2) * np.sum(ww ** 2) + noise_var) / (r ** 2 * W ** 2))
            out["m_pred"] = float(erf(1.0 / np.sqrt(2.0 * out["rho_eff"])))
        else:
            out["M_eff"] = float(mem.M)
            out["rho_eff"] = rho_of(r, mem.M)
            out["rho_eff0"] = out["rho_eff"]
            out["m_pred"] = float(erf(1.0 / np.sqrt(2.0 * out["rho_eff"])))
    else:
        # pairwise: predicted class = class with largest summed overlap
        over = mem.keys @ sigma / N
        cls = np.bincount(mem.labels, weights=over ** 2, minlength=mem.K)
        out["pred"] = int(np.argmax(cls))
        out["M_eff"] = float(mem.M)
        out["rho_eff"] = rho_of(r, mem.M)
        out["rho_eff0"] = out["rho_eff"]
        out["m_pred"] = float(erf(1.0 / np.sqrt(2.0 * out["rho_eff"])))
    return out


def rho_of(r, M):
    return (1 - r ** 2) / (M * r ** 2)


def r_of(rho, M):
    return 1.0 / np.sqrt(1 + rho * M)


def m_majority(rho):
    """One-step magnetization from the majority rule over M pooled examples
    (EPL 2023, SM eq. 16): m = erf(1/sqrt(2 rho))."""
    rho = np.asarray(rho, dtype=float)
    return erf(1.0 / np.sqrt(2.0 * np.maximum(rho, 1e-12)))


def pairwise_sup_boundary(rho):
    """EPL 2023 signal-to-noise retrieval condition alpha(1+rho)^2 + rho < 1 -> alpha_c(rho)."""
    return (1 - rho) / (1 + rho) ** 2


# ------------------------------------------------------------------- experiment core
def trial(N, K, M, r, model, lam, rng, T=10, n_queries=4, r_test=None):
    """One dataset realisation; average diagnostics over n_queries fresh test examples."""
    r_test = r if r_test is None else r_test
    xi, eta = make_dataset(N, K, M, r, rng)
    mem = HebbianMemory(xi, eta, model, lam)
    acc = []
    for q in range(n_queries):
        mu = int(rng.integers(K))
        s0 = noisy_copy(xi[mu], r_test, rng).astype(np.float32)
        s1, sT, w = mem.run(s0, T=T)
        d1 = diagnostics(mem, s1, mu, w, r)      # one attention step (a transformer layer)
        dT = diagnostics(mem, sT, mu, w, r)      # iterated to a fixed point (Hopfield retrieval)
        d = {k + "_1": v for k, v in d1.items()}
        d.update({k + "_T": v for k, v in dT.items()})
        d["correct"] = float(d1["pred"] == mu)
        acc.append(d)
    keys = acc[0].keys()
    return {k: float(np.mean([a[k] for a in acc])) for k in keys}
