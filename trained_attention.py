"""
trained_attention.py — an attention head of the form of Definition 3 (keys = values, softmax weights) whose keys are
trained by gradient descent, without labels, on the noisy-example dataset of EPL 2023, compared with the Hebbian
attention (keys = examples) and with the majority-rule value erf(1/sqrt(2 rho)).

model:      h(s) = sum_b softmax_b( beta k_b . s ) k_b,   H = K M learnable slots k_b (one per example), learnable beta
objective:  L = (1/(K M N)) sum_{mu,a} || h(eta^{mu a} * chi') - eta^{mu a} ||^2,  P(chi'_i = +-1) = (1 +- r')/2,
            a fresh corruption chi' at every step (denoising; no labels, archetypes or prototypes enter training).
Pure numpy with hand-written backprop + Adam.  On fresh test examples we measure the overlap m^(1) of sgn h with the
archetype, what the slots have learnt (largest cosine with archetypes and with training examples) and how many slots
carry the weight of a query.
"""
import sys, os, time, json
import numpy as np
from scipy.special import erf
from hebbattn import make_dataset, noisy_copy, rho_of

QUICK = "--quick" in sys.argv
os.makedirs("results", exist_ok=True)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


class DenoisingHead:
    def __init__(self, N, H, rng, beta0):
        self.N = N
        self.K = rng.standard_normal((H, N)).astype(np.float64) / np.sqrt(N)   # slots, used as keys and values
        self.b = np.log(beta0)
        self.params = ["K", "b"]
        self.m = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in self.params}
        self.v = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in self.params}
        self.t = 0

    def forward(self, X):
        Z = X @ self.K.T
        S = softmax(np.exp(self.b) * Z)
        return Z, S, S @ self.K

    def loss_grad(self, X, T):
        B = X.shape[0]; beta = np.exp(self.b)
        Z, S, Hh = self.forward(X)
        R = Hh - T
        loss = np.sum(R ** 2) / (B * self.N)
        dH = 2 * R / (B * self.N)
        dS = dH @ self.K.T
        G = S * (dS - (dS * S).sum(1, keepdims=True))                         # grad wrt beta*Z
        dK = S.T @ dH + beta * G.T @ X                                        # through the values and through the keys
        return loss, dict(K=dK, b=float(np.sum(G * Z) * beta))

    def adam_step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            self.m[p] = b1 * self.m[p] + (1 - b1) * g[p]
            self.v[p] = b2 * self.v[p] + (1 - b2) * g[p] ** 2
            mh = self.m[p] / (1 - b1 ** self.t); vh = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + eps))


def run_one(N, Kc, M, r, H, r_corr, rng, crng, steps, n_test=400):
    xi, eta = make_dataset(N, Kc, M, r, rng)
    Xtr = eta.reshape(Kc * M, N).astype(np.float64)
    yte = rng.integers(Kc, size=n_test)
    Xte = np.stack([noisy_copy(xi[c], r, rng) for c in yte]).astype(np.float64)
    xif = xi.astype(np.float64)
    net = DenoisingHead(N, H, rng, beta0=1.0 / np.sqrt(N))
    for t in range(steps):
        Xin = Xtr * np.where(crng.random(Xtr.shape) < (1 + r_corr) / 2, 1.0, -1.0)   # fresh corruption of every example
        _, g = net.loss_grad(Xin, Xtr)
        net.adam_step(g, lr=0.015 if t < steps // 2 else 0.005)
    _, S, Hh = net.forward(Xte)
    m = float(np.mean(np.sign(Hh) * xif[yte]))
    # what did the slots learn? largest cosine of each slot with an archetype and with a training example,
    # averaged over the slots with weights given by their mean attention on the test queries
    Kn = net.K / (np.linalg.norm(net.K, axis=1, keepdims=True) + 1e-12)
    usage = S.mean(0)
    cos_arch = float(np.average(np.abs(Kn @ xif.T / np.sqrt(N)).max(1), weights=usage))
    cos_ex = float(np.average(np.abs(Kn @ Xtr.T / np.sqrt(N)).max(1), weights=usage))
    slots = float(np.mean(1 / np.sum(S ** 2, 1)))                             # slots carrying the weight of a query
    # inverse temperature of the paper's head that gives the same logits: beta |k| / sqrt(N) (keys = examples have norm sqrt(N))
    beta_eff = float(np.exp(net.b) * np.average(np.linalg.norm(net.K, axis=1), weights=usage) / np.sqrt(N))
    # Hebbian reference on the same data: one step of Eq. (attn) with keys = training examples, best beta on a grid
    grid = np.logspace(-3, 0, 25)
    mg = [float(np.mean(np.sign(softmax(lam * (Xte @ Xtr.T)) @ Xtr) * xif[yte])) for lam in grid]
    # the trained slots read out with a second inverse temperature beta_2 set by hand, in the units of the paper's head,
    # chosen on the same grid and with the same criterion as the Hebbian beta
    knorm = np.average(np.linalg.norm(net.K, axis=1), weights=usage)
    m2 = [float(np.mean(np.sign(softmax(lam * np.sqrt(N) / knorm * (Xte @ net.K.T)) @ net.K) * xif[yte])) for lam in grid]
    return dict(m=m, m_beta2=max(m2), beta2=float(grid[int(np.argmax(m2))]), m_hebb=max(mg), beta_hebb=float(grid[int(np.argmax(mg))]),
                cos_arch=cos_arch, cos_ex=cos_ex, ratio=cos_arch / cos_ex, slots=slots, beta=float(np.exp(net.b)), beta_eff=beta_eff)


def main():
    N, Kc = (300, 15) if QUICK else (400, 20)
    r_corr = 0.5
    steps = 500 if QUICK else 1000
    Ms = [4, 16, 64]
    rs = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    seeds = 1 if QUICK else 2
    rows = []
    t0 = time.time()
    for M in Ms:
        for r in rs:
            H = Kc * M                                                  # as many slots as training examples
            rr = [run_one(N, Kc, M, r, H, r_corr, np.random.default_rng(100 * M + int(100 * r) + s),
                          np.random.default_rng(10_000 + s), steps) for s in range(seeds)]
            rho = rho_of(r, M)
            row = dict(M=M, r=r, rho=rho, m_major=float(erf(1 / np.sqrt(2 * rho))),
                       **{k: float(np.mean([x[k] for x in rr])) for k in rr[0]})
            rows.append(row)
            print(f"M={M:3d} r={r:.2f} rho={rho:.3f} m={row['m']:.3f} hebb={row['m_hebb']:.3f} majority={row['m_major']:.3f} "
                  f"m(beta_2)={row['m_beta2']:.3f} beta_2*N={row['beta2']*N:.1f} beta_eff*N={row['beta_eff']*N:.1f} beta_hebb*N={row['beta_hebb']*N:.1f} cos_arch={row['cos_arch']:.2f} cos_ex={row['cos_ex']:.2f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open("results/E5_trained_attention.json", "w"))


if __name__ == "__main__":
    main()
