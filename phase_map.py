"""phase_map.py — regime map of one attention step in the (beta, rho) plane from the closed-form theory."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import paper_figs as PF
from theory import one_step, phi_M, lam_edges
from scipy.optimize import minimize_scalar
from hebbattn import r_of

N, K, M = 1000, 100, 16
betas = np.logspace(-3, 0, 240)
rhos = np.linspace(0.005, 2.0, 220)
REG = np.zeros((len(rhos), len(betas)), int); M1 = np.zeros_like(REG, dtype=float)
xc = np.sqrt(2 * np.log(M))
for i, rh in enumerate(rhos):
    r = r_of(rh, M)
    th = one_step(betas, N, K, M, r)
    x = betas * np.sqrt(N * (1 - r ** 2))
    ct = (1 + rh) * (1 - th["W"]) ** 2 / ((K - 1) * th["W"] ** 2)           # crosstalk term of Corollary 1(i)
    reg = np.where(ct > rh, 0, np.where(x > xc, 2, 1))
    REG[i] = reg; M1[i] = th["m1"]
# analytic edges (Eq. 17) and position of the maximum (Remark 4)
rr = r_of(rhos, M)
b_lo, b_hi = lam_edges(N, K, M, rr)
def argmax_m1(r):                                                        # maximum of the full expression (15)
    lo, hi = lam_edges(N, K, M, r)
    f = lambda lb: -one_step(np.array([np.exp(lb)]), N, K, M, r)["m1"][0]
    return np.exp(minimize_scalar(f, bounds=(np.log(0.5 * lo), np.log(min(3 * hi, 1.0))), method="bounded", options={"xatol": 1e-4}).x)
b_max = np.array([argmax_m1(r) for r in rr])

cols = [PF.COL[1], PF.COL[0], PF.COL[2]]
cmap = ListedColormap([to_rgba(c, 0.55) for c in cols])
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
ext = [np.log10(betas[0]), np.log10(betas[-1]), rhos[0], rhos[-1]]
ax = axes[0]
ax.imshow(REG, origin="lower", aspect="auto", extent=ext, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
for a in axes:
    a.plot(np.log10(b_lo), rhos, color="k", lw=1.0, label=r"$\beta_{\rm lo}$: crosstalk term $=\rho$")
    a.plot(np.log10(b_hi), rhos, color="k", lw=1.0, ls=(0, (6, 3)), label=r"$\beta_{\rm hi}$: $x=\sqrt{2\ln M}$")
    ok = (b_max < b_hi) & (rhos > 0.02)                                  # the closure is unreliable beyond the condensation edge; flat top below rho=0.02
    a.plot(np.log10(b_max[ok]), rhos[ok], color="k", lw=1.2, ls=(0, (1, 2)), label=r"$\beta^*$: maximum of $m^{(1)}$")
    a.set_xlabel(r"$\beta$"); a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
    ticks = [-3, -2, -1, 0]; a.set_xticks(ticks); a.set_xticklabels([r"$10^{%d}$" % t for t in ticks])
ax.set_ylabel(r"$\rho$")
ax.text(-2.55, 0.9, "crosstalk", ha="center", fontsize=9, color="k")
ax.text(-1.55, 0.18, "prototype", ha="center", fontsize=9, color="k")
ax.text(-0.45, 0.8, "memorization", ha="center", fontsize=9, color="k")
ax.set_title(r"(a) regimes, $N=%d$, $K=%d$, $M=%d$" % (N, K, M), fontsize=8.5, loc="left")
ax2 = axes[1]
im = ax2.imshow(M1, origin="lower", aspect="auto", extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="bilinear")
cs = ax2.contour(np.log10(betas), rhos, M1, levels=[0.5, 0.7, 0.9], colors="0.25", linewidths=0.6)
ax2.clabel(cs, fmt="%.1f", fontsize=7, inline=True)
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03); cb.set_label(r"$m^{(1)}$ (theory)"); cb.ax.tick_params(labelsize=7)
ax2.set_title(r"(b) one-step magnetization, theory", fontsize=8.5, loc="left")
ax2.legend(loc="upper left", fontsize=7, handlelength=1.6, framealpha=0.95)
fig.tight_layout(w_pad=0.6)
fig.savefig("paper/figs/fig_phase.pdf")
print("window closes at rho =", rhos[np.argmax(b_lo > b_hi)] if np.any(b_lo > b_hi) else None); print("beta*/beta_lo at rho=0.2:", np.interp(0.2, rhos, b_max / b_lo))
