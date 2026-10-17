"""paper_figs.py — manuscript figures (LaTeX look: Latin Modern + Computer Modern math).
Theory = solid lines, Monte Carlo = symbols with standard errors.   python3 paper_figs.py
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.interpolate import griddata
from theory import one_step, phi_M, lam_edges
from hebbattn import r_of, m_majority
from scipy.special import erf

for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "lmroman10-*.otf")):   # Latin Modern, in the repo
    fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "axes.grid": False,
    "lines.linewidth": 1.3, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "errorbar.capsize": 1.5, "hatch.linewidth": 0.5})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#eda100"]
MK = ["o", "s", "^", "D", "v", "p"]
os.makedirs("paper/figs", exist_ok=True)
J = lambda n: json.load(open(f"results/{n}.json"))


def mc(ax, x, y, se, color, marker, label=None, ms=3.2, mfc=None, **kw):
    ax.errorbar(x, y, yerr=se, fmt=marker, color=color, ms=ms, mfc=color if mfc is None else mfc,
                mew=0.7, elinewidth=0.6, label=label, zorder=3, **kw)


# =============================================================================== Fig. 4 (the load disappears)
def fig1():
    Sa = J("paper_S1a_N4000"); Sb = J("paper_S1b_N4000")
    N, M = Sa[0]["N"], Sa[0]["M"]
    beta = [x["lam"] for x in Sa if x["model"] == "attn_unsup"][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    # (a) m^(1) vs alpha at three rho, pairwise vs attention
    ax = axes[0]
    A = np.logspace(np.log10(0.01), np.log10(0.5), 120)
    rhos_a = sorted({x["rho"] for x in Sa})
    for k, rh in enumerate(rhos_a):
        r = r_of(rh, M)
        ax.plot(A, erf(1 / np.sqrt(2 * rh + 2 * A * (1 + rh) ** 2 / r ** 2)), color=COL[k], ls="--")
        ax.plot(A, [one_step(beta, N, max(2, int(round(a * N))), M, r)["m1"][0] for a in A], color=COL[k])
        for model, mk, mfc in [("pairwise_sup", "s", "white"), ("attn_unsup", "o", None)]:
            rows = sorted([x for x in Sa if x["model"] == model and x["rho"] == rh], key=lambda x: x["alpha"])
            mc(ax, [x["alpha"] for x in rows], [x["m_1"] for x in rows], [x["m_1_se"] for x in rows], COL[k], mk,
               r"$\rho=%g$" % rh if model == "attn_unsup" else None, mfc=mfc)
    ax.set_xscale("log"); ax.set_xlabel(r"$\alpha=K/N$"); ax.set_ylabel(r"$m^{(1)}$"); ax.set_ylim(0.15, 1.03)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3]); ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3"]); ax.minorticks_off()
    ax.set_title(r"(a) one step at fixed $\rho$, $\beta=%g$" % beta, fontsize=8.5, loc="left")
    ax.legend(loc="lower left", handlelength=1.2, labelspacing=0.15)
    # (b) m vs rho at fixed alpha: pairwise, attention one step, attention iterated
    ax = axes[1]
    a0 = Sb[0]["alpha"]; K = int(a0 * N)
    R = np.linspace(0.005, 0.8, 200); rr = r_of(R, M)
    ax.plot(R, erf(1 / np.sqrt(2 * R + 2 * a0 * (1 + R) ** 2 / rr ** 2)), color=COL[1], ls="--")
    ax.plot(R, [one_step(beta, N, K, M, r_of(rh, M))["m1"][0] for rh in R], color=COL[0])
    ax.plot(R, rr, color="0.45", lw=0.7, ls=":")
    ax.text(0.62, r_of(0.62, M) + 0.04, r"$r$", color="0.3", fontsize=8)
    pw = sorted([x for x in Sb if x["model"] == "pairwise_sup"], key=lambda x: x["rho"])
    at = sorted([x for x in Sb if x["model"] == "attn_unsup"], key=lambda x: x["rho"])
    mc(ax, [x["rho"] for x in pw], [x["m_1"] for x in pw], [x["m_1_se"] for x in pw], COL[1], "s", "pairwise, one step", mfc="white")
    mc(ax, [x["rho"] for x in at], [x["m_1"] for x in at], [x["m_1_se"] for x in at], COL[0], "o", "attention, one step")
    mc(ax, [x["rho"] for x in at], [x["m_T"] for x in at], [x["m_T_se"] for x in at], COL[2], "^", "attention, fixed point")
    ax.set_xlabel(r"$\rho=(1-r^2)/(Mr^2)$"); ax.set_ylabel(r"$m$"); ax.set_ylim(0.15, 1.03); ax.set_xlim(0, 0.8)
    ax.set_title(r"(b) fixed load $\alpha=%g$" % a0, fontsize=8.5, loc="left"); ax.legend(loc="upper right", handlelength=1.2, labelspacing=0.15)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig("paper/figs/fig_maps.pdf"); plt.close(fig)


# =============================================================================== Fig. 3 (temperature scan)
def fig2():
    S2 = J("paper_S2_lambda")
    cfg = sorted({(x["rho"], x["M"]) for x in S2}, key=lambda t: t[0])
    N = S2[0]["N"]; K = int(0.1 * N)
    L = np.logspace(-2.8, 0, 300)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0), sharex=True)
    for j, (rh, M) in enumerate(cfg):
        r = r_of(rh, M)
        rows = sorted([x for x in S2 if x["rho"] == rh and x["M"] == M], key=lambda x: x["lam"])
        lam = np.array([x["lam"] for x in rows])
        th = one_step(L, N, K, M, r)
        lo, hi = lam_edges(N, K, M, r)
        for row in axes[:, j]:
            row.axvspan(lo, hi, facecolor="0.5", edgecolor="0.5", hatch="//", lw=0, zorder=0, alpha=0.18)
        ax = axes[0, j]
        ax.plot(L, th["m1"], color=COL[0], label=r"theory $m^{(1)}$")
        mc(ax, lam, [x["m_1"] for x in rows], [x["m_1_se"] for x in rows], COL[0], "o", r"MC $m^{(1)}$")
        mc(ax, lam, [x["n_max_same_1"] for x in rows], [x["n_max_same_1_se"] for x in rows], COL[1], "s",
           r"MC $n_{\max}$", mfc="white")
        ax.axhline(r, color="0.4", lw=0.7, ls=":")
        ax.set_ylim(0, 1.05); ax.set_title(r"$\rho=%g,\ M=%d\ (r=%.2f)$" % (rh, M, r), fontsize=8.5)
        ax = axes[1, j]
        ax.plot(L, th["W"], color=COL[2], label=r"theory $W$")
        ax.plot(L, th["M_eff"] / M, color=COL[3], label=r"theory $M_{\rm eff}/M$")
        mc(ax, lam, [x["w_true_class_1"] for x in rows], [x["w_true_class_1_se"] for x in rows], COL[2], "^", r"MC $W$")
        mc(ax, lam, [x["M_eff_1"] / M for x in rows], [x["M_eff_1_se"] / M for x in rows], COL[3], "D", r"MC $M_{\rm eff}/M$")
        ax.set_ylim(0, 1.05); ax.set_xscale("log"); ax.set_xlabel(r"$\beta$"); ax.set_xlim(1.2e-3, 1.1)
    axes[0, 0].set_ylabel(r"overlap"); axes[1, 0].set_ylabel(r"attention statistics")
    h1, l1 = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07), handlelength=1.2, fontsize=7, columnspacing=1.0)
    fig.savefig("paper/figs/fig_lambda.pdf"); plt.close(fig)


# =============================================================================== Fig. 5 (size dependence)
def fig3():
    S3 = J("paper_S3_N")
    Ns = sorted({x["N"] for x in S3}); rh, M = S3[0]["rho"], S3[0]["M"]; r = r_of(rh, M)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
    L = np.logspace(-2.8, 0.1, 300)
    for k, N in enumerate(Ns):
        rows = sorted([x for x in S3 if x["N"] == N], key=lambda x: x["lam"])
        lam = np.array([x["lam"] for x in rows]); K = int(0.1 * N)
        th = one_step(L, N, K, M, r)
        for ax, sc in zip(axes, [1.0, N, np.sqrt(N)]):
            ax.plot(L * sc, th["m1"], color=COL[k])
            mc(ax, lam * sc, [x["m_1"] for x in rows], [x["m_1_se"] for x in rows], COL[k], MK[k], r"$N=%d$" % N)
    for ax, lab in zip(axes, [r"$\beta$", r"$\beta N$", r"$\beta\sqrt{N}$"]):
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylim(0, 1.05)
    axes[0].set_xlim(1.2e-3, 1.2); axes[1].set_xlim(0.5, 1.5e3); axes[2].set_xlim(0.02, 40)
    axes[0].set_ylabel(r"$m^{(1)}$"); axes[2].legend(loc="lower right", handlelength=1.2, labelspacing=0.15)
    axes[0].set_title(r"(a) $\rho=%g,\ M=%d,\ \alpha=0.1$" % (rh, M), fontsize=8.5, loc="left")
    axes[1].set_title(r"(b) crosstalk edge: $\beta N r^2\sim\ln K$", fontsize=8.5, loc="left")
    axes[2].set_title(r"(c) condensation edge: $\beta\sqrt{N(1-r^2)}\sim\sqrt{2\ln M}$", fontsize=8.5, loc="left")
    fig.savefig("paper/figs/fig_N.pdf"); plt.close(fig)


# =============================================================================== Fig. 6 (REM participation ratio)
def fig4():
    S4 = J("paper_S4_rem")
    cfg = []
    for x in S4:
        t = (x["N"], x["M"], x["r"])
        if t not in cfg: cfg.append(t)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    X = np.logspace(-1, 1.2, 200)
    done = set()
    for k, (N, M, r) in enumerate(cfg):
        rows = sorted([x for x in S4 if (x["N"], x["M"], x["r"]) == (N, M, r)], key=lambda x: x["lam"])
        if M not in done:
            ax.plot(X, phi_M(X, M), color=COL[k], label=r"$\Phi_{%d}(x)$" % M)
            ax.axvline(np.sqrt(2 * np.log(M)), color=COL[k], lw=0.6, ls=":")
            done.add(M)
        col = COL[[c[1] for c in cfg].index(M)]
        xs_ = [x["lam"] * np.sqrt(N * (1 - r ** 2)) for x in rows]
        mc(ax, xs_, [x["Meff_over_M"] for x in rows], [x["Meff_over_M_se"] for x in rows], col, MK[k],
           (r"MC $N=%d,\,M=%d$" % (N, M)) + (r", $r=%g$" % r if r != 0.5 else ""), mfc="white" if k >= 3 else None)
    ax.set_xscale("log"); ax.set_xlabel(r"$x=\beta\sqrt{N(1-r^2)}$"); ax.set_ylabel(r"$M_{\rm eff}/M$")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6.3, handlelength=1.1, labelspacing=0.12, loc="lower left", borderpad=0.3)
    fig.savefig("paper/figs/fig_rem.pdf"); plt.close(fig)


# =============================================================================== Fig. 7 (collapse on rho_eff)
def fig5():
    pts = [x for x in J("paper_S1_maps") if x["model"] == "attn_unsup"] + J("paper_S2_lambda") + J("paper_S3_N")
    pw = [x for x in J("paper_S1_maps") if x["model"] == "pairwise_sup"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lo = 2e-3
    x = np.clip([p["rho_eff_1"] for p in pts], lo, 60); y = [p["m_1"] for p in pts]
    c = [p["M_eff_1"] / p["M"] for p in pts]
    sc = ax.scatter(x, y, c=c, cmap="Blues", s=7, edgecolor="0.3", linewidth=0.25, vmin=0, vmax=1, zorder=3)
    ax.scatter(np.clip([p["rho"] for p in pw], lo, 60), [p["m_1"] for p in pw], marker="x", color=COL[1], s=9, lw=0.6,
               label=r"pairwise Hebb, against bare $\rho$", zorder=2)
    xx = np.logspace(np.log10(lo), 1.8, 300)
    ax.plot(xx, m_majority(xx), color="k", lw=1.0, label=r"$\mathrm{erf}(1/\sqrt{2\rho_{\rm eff}})$", zorder=4)
    ax.set_xscale("log"); ax.set_xlim(lo * 0.8, 70); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\rho_{\rm eff}$ (attention); bare $\rho$ for the pairwise crosses"); ax.set_ylabel(r"$m^{(1)}$")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03); cb.set_label(r"$M_{\rm eff}/M$"); cb.ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", handlelength=1.4)
    fig.savefig("paper/figs/fig_collapse.pdf"); plt.close(fig)


# =============================================================================== Fig. 9 (gradient descent without labels)
def fig6():
    from matplotlib.lines import Line2D
    E5 = J("E5_trained_attention"); N, Kc = 400, 20
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75)); fig.subplots_adjust(wspace=0.45, bottom=0.27)
    Ms = sorted({x["M"] for x in E5})
    # one encoding in (a) and (b): filled = trained head at its learned beta, x = trained head at a hand-set beta_2,
    # open = Hebbian head at its best beta; colour (and shape) = M
    sty_l = dict(ms=3.6, mec="k", mew=0.3)                      # learned
    sty_2 = dict(marker="x", ms=4.2, mew=1.0, lw=0)              # beta_2
    sty_h = dict(ms=6.0, mfc="none", mew=0.9)                    # Hebbian
    rr = np.logspace(-2.1, 0.85, 80)
    axes[0].plot(rr, m_majority(rr), color="0.4", lw=0.8, ls=":")
    ax = axes[1]
    for k, M in enumerate(Ms):
        sub = sorted([x for x in E5 if x["M"] == M], key=lambda x: x["rho"])
        X = [x["rho"] for x in sub]
        axes[0].plot(X, [x["m_hebb"] for x in sub], color=COL[k], lw=1.0, marker=MK[k], **sty_h)
        axes[0].plot(X, [x["m"] for x in sub], color=COL[k], lw=1.0, ls="--", marker=MK[k], **sty_l)
        axes[0].plot(X, [x["m_beta2"] for x in sub], color=COL[k], **sty_2)
        rho_g = np.logspace(-2.3, 1.0, 120); lo, hi = lam_edges(N, Kc, M, r_of(rho_g, M))
        ax.plot(lo, rho_g, color=COL[k], lw=0.8); ax.plot(hi, rho_g, color=COL[k], lw=0.8, ls="--")
        ok = lo < hi; ax.fill_betweenx(rho_g[ok], lo[ok], hi[ok], color=COL[k], alpha=0.08, lw=0)
        ax.plot([x["beta_hebb"] for x in sub], X, MK[k], color=COL[k], lw=0, **sty_h, zorder=4)
        ax.plot([x["beta2"] for x in sub], X, color=COL[k], **sty_2, zorder=5)
        ax.plot([x["beta_eff"] for x in sub], X, MK[k], color=COL[k], lw=0, **sty_l, zorder=5)
        axes[2].plot(X, [x["cos_arch"] for x in sub], color=COL[k], lw=1.0, marker=MK[k], ms=3)
        axes[2].plot(X, [x["cos_ex"] for x in sub], color=COL[k], lw=1.0, ls=":", marker=MK[k], ms=3, mfc="none")
    axes[0].set_xscale("log"); axes[0].set_xlabel(r"$\rho$"); axes[0].set_ylim(0, 1.03); axes[0].set_ylabel(r"$m^{(1)}$ on test examples")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4e-3, 0.5); ax.set_ylim(5e-3, 10)
    ax.text(0.0045, 0.03, "cross-\ntalk", fontsize=6.3, color="0.3"); ax.text(0.19, 0.0115, "memo-\nrization", fontsize=6.0, color="0.3")
    ax.set_xlabel(r"inverse temperature $\beta$"); ax.set_ylabel(r"$\rho$")
    ax.legend(handles=[Line2D([], [], color="0.3", lw=0.8, label="crosstalk edge"), Line2D([], [], color="0.3", lw=0.8, ls="--", label="condensation edge")],
              fontsize=6.0, loc="upper left", handlelength=1.8, labelspacing=0.25)
    axes[2].set_xscale("log"); axes[2].set_xlabel(r"$\rho$"); axes[2].set_ylim(0, 1.03); axes[2].set_ylabel(r"$\max|\cos|$ of learned slots")
    axes[2].legend(handles=[Line2D([], [], color="0.3", lw=1.0, marker="o", ms=3, label="with archetypes"),
                            Line2D([], [], color="0.3", lw=1.0, ls=":", marker="o", ms=3, mfc="none", label="with examples")],
                   fontsize=6.0, loc="lower left", labelspacing=0.25)
    for j, t in enumerate("abc"): axes[j].set_title("(%s)" % t, fontsize=8.5, loc="left")
    # shared legend below the panels
    hs = [Line2D([], [], color="0.3", lw=1.0, ls="--", marker="o", **sty_l, label=r"trained, learned $\beta$"),
          Line2D([], [], color="0.3", **sty_2, label=r"trained, hand-set $\beta_2$"),
          Line2D([], [], color="0.3", lw=1.0, marker="o", **sty_h, label=r"Hebbian, best $\beta$"),
          Line2D([], [], color="0.4", lw=0.8, ls=":", label="majority rule")] + \
         [Line2D([], [], color=COL[k], lw=0, marker=MK[k], ms=4, label=r"$M=%d$" % M) for k, M in enumerate(Ms)]
    fig.legend(handles=hs, loc="lower center", ncol=7, fontsize=6.5, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig("paper/figs/fig_gd.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]:
        f(); print(f.__name__, "ok", flush=True)
