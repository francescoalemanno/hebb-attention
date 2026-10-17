"""regimes_fig.py — Fig. 1: the three regimes as the softmax weights of one query over the KM keys.
Each panel: one inverse temperature; bars = attention weights of the keys, grouped by class, with the
correct class first (filled) and the other classes in grey; annotations give W, M_eff and the outcome of the update."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_figs as PF
from hebbattn import make_dataset, noisy_copy, HebbianMemory, r_of
from theory import lam_edges

N, K, M, rho = 1000, 20, 16, 0.2
r = r_of(rho, M)
rng = np.random.default_rng(3)
xi, eta = make_dataset(N, K, M, r, rng)
mu = 0
s0 = noisy_copy(xi[mu], r, rng).astype(np.float32)
lo, hi = lam_edges(N, K, M, r)
betas = [lo / 3, np.sqrt(lo * hi), hi * 3]
titles = ["(a) crosstalk", "(b) prototype", "(c) memorization"]
fig = plt.figure(figsize=(7.1, 2.4))
outer = fig.add_gridspec(1, 3, wspace=0.22)
ax0 = None
for j, (b, title) in enumerate(zip(betas, titles)):
    inner = outer[j].subgridspec(1, 2, width_ratios=[M, K - 1], wspace=0.08)
    axL = fig.add_subplot(inner[0], sharey=ax0); axR = fig.add_subplot(inner[1], sharey=axL)
    if ax0 is None: ax0 = axL
    mem = HebbianMemory(xi, eta, "attn_unsup", b)
    h, w = mem.field(s0)
    s1 = np.sign(h)
    m1 = float(xi[mu] @ s1 / N)
    same = mem.labels == mu
    W = w[same].sum(); Meff = W ** 2 / np.sum(w[same] ** 2)
    nmax = float((eta[mu].astype(np.float32) @ s1 / N).max())
    V = np.bincount(mem.labels, weights=w, minlength=K)[1:]          # total weight of each wrong class
    axL.bar(np.arange(M), w[same], width=0.8, color=PF.COL[0])
    axR.bar(np.arange(K - 1), V, width=0.8, color="0.55")
    axL.set_yscale("log"); axL.set_ylim(1e-5, 3)
    axL.set_xticks([]); axR.set_xticks([]); plt.setp(axR.get_yticklabels(), visible=False)
    if j > 0: plt.setp(axL.get_yticklabels(), visible=False)
    if V.max() < 1e-5:
        axR.text(0.5, 0.08, r"all $V_\nu<10^{-25}$", transform=axR.transAxes, ha="center", fontsize=7, color="0.3")
    axL.set_xlabel(r"class 1: $w_b$", fontsize=8); axR.set_xlabel(r"classes $2..K$: $V_\nu$", fontsize=8)
    axL.set_title(r"%s, $\beta=%.3f$" % (title, b), fontsize=8.5, loc="left")
    axR.text(0.96, 0.95, r"$W=%.2f$" "\n" r"$M_{\rm eff}=%.1f$" "\n" r"$m^{(1)}=%.2f$" "\n" r"$n_{\max}=%.2f$" % (W, Meff, m1, nmax),
             transform=axR.transAxes, ha="right", va="top", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    if j == 0:
        axL.set_ylabel(r"attention weight")
fig.savefig("paper/figs/fig_regimes.pdf")
print("betas", [round(x, 4) for x in betas], "edges", round(lo, 4), round(hi, 4), "r", round(r, 2))
