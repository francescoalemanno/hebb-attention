"""
theory.py — closed-form signal-to-noise theory for one attention step over a supervised
Hebbian dataset (no input from simulations).

Setting: K archetypes, M examples/class of quality r, N neurons, query = fresh example of
quality q (q=1: the archetype itself). Logit of key x: lam*N*m_x(sigma).

Conditionally on the query (whose own overlap with the archetype is q), the key overlaps are
    true class:    q r + z_a sqrt((1-r^2)/N)                 (common shift q r, iid spread)
    class nu != 1: r g_nu + z_b sqrt((1-r^2)/N),  g_nu ~ N(0,1/N) (class-level shift shared by its M keys)
Common shifts cancel inside each class's softmax, so the within-class REM variable is
    x = lam sqrt(N (1-r^2)),  independent of q.

Annealed weight of the true class:
    W(lam) = [1 + (K-1) exp(-N lam q r + N lam^2 r^2 / 2)]^{-1}
    (ratio of the annealed within-class sum  M exp(N lam q r + lam^2 N (1-r^2)/2)
     and the annealed other-class sum (K-1) M exp(lam^2 N r^2/2 + lam^2 N (1-r^2)/2));
    valid while the class-level spread lam r sqrt(N) < sqrt(2 ln K) and x < sqrt(2 ln M).

Participation ratio of a softmax of M iid Gaussian logits with std x (Random Energy Model):
    Phi_M(x) = E[ (sum_a e^{x z_a})^2 / sum_a e^{2 x z_a} ] / M,   condenses at x ~ sqrt(2 ln M).
    M_eff = M'_eff = M Phi_M(lam sqrt(N(1-r^2))).

Readout X_i = xi_i h_i:  mean  r W + tanh(lam) q           (self-coupling from the i-th logit factor)
                         var   (1-r^2) W^2/M_eff + (1-W)^2/(K-1) [r^2 + (1-r^2)/M'_eff] + tanh(lam)^2 (1-q^2)
    m^{(1)} = erf( mean / sqrt(2 var) )  = erf( 1/sqrt(2 rho_eff) ),  rho_eff = var/mean^2.
"""
import numpy as np
from scipy.special import erf

_rng = np.random.default_rng(12345)
_Z = _rng.standard_normal((4000, 512))  # cached Gaussian logits for Phi_M


def phi_M(x, M):
    """REM participation ratio / M for M iid N(0,1) logits scaled by x (vectorised in x)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    z = _Z[:, :M]
    out = np.empty_like(x)
    for k, xv in enumerate(x):
        a = xv * (z - z.max(1, keepdims=True))
        e = np.exp(a)
        out[k] = np.mean(e.sum(1) ** 2 / (e ** 2).sum(1)) / M
    return out


def class_weight(lam, N, K, r, q=1.0):
    lam = np.asarray(lam, dtype=float)
    expo = -N * lam * q * r + 0.5 * N * lam ** 2 * r ** 2
    return 1.0 / (1.0 + (K - 1) * np.exp(np.clip(expo, -700, 700)))


def one_step(lam, N, K, M, r, q=None):
    """Returns dict with W, M_eff, M_eff_other, rho_eff, m1 for an array of lam."""
    q = r if q is None else q
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    W = class_weight(lam, N, K, r, q)
    Meff = M * phi_M(lam * np.sqrt(N * (1 - r ** 2)), M)
    Meff_o = Meff
    t = np.tanh(lam)
    mean = r * W + t * q
    var = (1 - r ** 2) * W ** 2 / Meff + (1 - W) ** 2 / max(K - 1, 1) * (r ** 2 + (1 - r ** 2) / Meff_o) \
        + t ** 2 * (1 - q ** 2)
    rho_eff = var / mean ** 2
    return dict(lam=lam, W=W, M_eff=Meff, M_eff_other=Meff_o, rho_eff=rho_eff,
                m1=erf(1.0 / np.sqrt(2.0 * rho_eff)))


def pairwise_one_step_from_archetype(alpha, rho):
    """EPL 2023, SM eq. (68)."""
    return erf(1.0 / np.sqrt(2 * alpha * (1 + rho) ** 2 + 2 * rho))


def lam_edges(N, K, M, r, q=None):
    """Analytic edges of the generalisation window.
    lo: crosstalk edge, where the crosstalk term (1+rho)(K-1) exp(-2 N lam q r) of rho_eff equals rho
        (the correct class then carries a weight W ~ K^{-1/2}); hi: condensation edge x = sqrt(2 ln M)."""
    q = r if q is None else q
    rho = (1 - r ** 2) / (M * r ** 2)
    lo = np.log((K - 1) * (1 + rho) / rho) / (2 * N * q * r)
    hi = np.sqrt(2 * np.log(M)) / np.sqrt(N * (1 - r ** 2))
    return lo, hi


def lam_star(N, K, M, r):
    """Position of the maximum of m^(1) (q = r), closed form (Remark 3 / Appendix C): balance between the
    decay of the crosstalk and the loss of pooling, (K-1) exp(-2 N lam r^2) = kappa lam, solved by Lambert W."""
    from scipy.special import lambertw
    rho = (1 - r ** 2) / (M * r ** 2)
    kappa = (M - 1) * rho ** 2 / (1 + rho)
    return float(np.real(lambertw(2 * N * r * r * (K - 1) / kappa)) / (2 * N * r * r))


def W_star(N, K, M, r):
    """Weight of the correct class at the maximum, (1-W*)/W* = sqrt(kappa (K-1) lam*)."""
    rho = (1 - r ** 2) / (M * r ** 2)
    kappa = (M - 1) * rho ** 2 / (1 + rho)
    return 1.0 / (1.0 + np.sqrt(kappa * (K - 1) * lam_star(N, K, M, r)))
