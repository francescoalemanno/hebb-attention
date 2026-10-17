"""mnist_fig.py — Fig. 8 (structured data) in the notation of the manuscript."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams['hatch.linewidth'] = 0.5
from matplotlib.lines import Line2D
import paper_figs as PF
from theory import phi_M
from mnist_attention import load, binarise

D = json.load(open('results/mnist_attention.json')); rows = D['rows']; refs = D['refs']
M = 256; names = [('mnist', 'MNIST', PF.COL[0], 'o'), ('fmnist', 'F-MNIST', PF.COL[1], 's')]
fig = plt.figure(figsize=(7.1, 3.85))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.3)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
ax = axes[0]; X = np.logspace(-1.3, 1.5, 100); ax.plot(X, phi_M(X, M), color='k', lw=1, label=r"$\Phi_{%d}(x)$, Prop. 3" % M)
for key, lab, c, mk in names:
    sub = sorted([r for r in rows if r['dataset'] == key and r['M'] == M], key=lambda r: r['beta'])
    ax.plot([r['x_emp'] for r in sub], [r['Meff_over_M'] for r in sub], mk, ms=3, color=c, label=lab)
ax.axvline(np.sqrt(2 * np.log(M)), color='0.5', lw=0.6, ls=':')
ax.set_xscale('log'); ax.set_xlabel(r"$x=\beta N\,\sigma_m$"); ax.set_ylabel(r"$M_{\rm eff}/M$"); ax.set_ylim(0, 1.03)
ax.set_title("(a) condensation of the softmax", fontsize=8.5, loc='left'); ax.legend(handlelength=1.3, loc='upper right')
ax = axes[1]; z0 = np.linspace(-4, 4, 200); ax.plot(z0, np.exp(-z0 ** 2 / 2) / np.sqrt(2 * np.pi), color='k', lw=1, label='Gaussian, Lemma 1')
for key, lab, c, mk in names:
    xtr, ytr = load(key, 'train'); xte, yte = load(key, 'test'); ztr, zte = binarise(xtr, xte); N = 784
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(ytr == cc)[0], M, replace=False) for cc in range(10)]); keys, labs = ztr[idx], ytr[idx]
    q = rng.choice(len(zte), 400, replace=False); Q, yq = zte[q], yte[q]; over = Q @ keys.T / N; same = labs[None, :] == yq[:, None]
    z = np.concatenate([(over[i, same[i]] - over[i, same[i]].mean()) / over[i, same[i]].std() for i in range(400)])
    h, e = np.histogram(z, bins=60, range=(-4, 4), density=True); ax.step(0.5 * (e[1:] + e[:-1]), h, where='mid', color=c, lw=1.2, label=lab)
ax.set_xlabel(r"$(m_{\eta^{\mu a}}-\bar m)/\sigma_m$"); ax.set_ylabel("density"); ax.set_xlim(-4, 4); ax.set_ylim(0, 0.6)
ax.set_title("(b) within-class overlaps", fontsize=8.5, loc='left'); ax.legend(handlelength=1.3, loc='upper left')
ax = axes[2]; N = 784; K = 10
for key, lab, c, mk in names:
    sub = sorted([r for r in rows if r['dataset'] == key and r['M'] == M], key=lambda r: r['beta'])
    ax.plot([r['beta'] for r in sub], [r['acc'] for r in sub], color=c, marker=mk, ms=2.5, lw=1.2, label=lab)
    ax.axhline(refs[f"{key}_M{M}"]['nc_acc'], color=c, ls=':', lw=0.8); ax.axhline(refs[f"{key}_M{M}"]['nn_acc'], color=c, ls='--', lw=0.8)
    # window (18) with the empirical overlap statistics in place of qr and sqrt((1-r^2)/N)
    ref = refs[f"{key}_M{M}"]
    rho_emp = N * ref['std_same'] ** 2 / (M * ref['mean_same'])          # rho = N sigma_m^2 / (M mbar) on Rademacher data
    b_lo = np.log((K - 1) * (1 + rho_emp) / rho_emp) / (2 * N * ref['mean_same']); b_hi = np.sqrt(2 * np.log(M)) / (N * ref['std_same'])
    ax.axvspan(b_lo, b_hi, facecolor=c, edgecolor=c, hatch='//' if key == 'mnist' else '\\\\', lw=0, alpha=0.18)
    best = max(sub, key=lambda r: r['acc'])
    ax.plot([best['beta']], [best['acc']], marker='*', ms=9, color=c, mec='k', mew=0.5, zorder=5)
    print(key, 'rho_emp', round(rho_emp, 3), 'window', round(b_lo, 4), round(b_hi, 4), 'best beta', round(best['beta'], 4), 'acc', round(best['acc'], 3))
h = [Line2D([], [], color=PF.COL[0], marker='o', ms=2.5, label='MNIST'), Line2D([], [], color=PF.COL[1], marker='s', ms=2.5, label='F-MNIST'),
     Line2D([], [], color='0.3', ls='--', lw=0.8, label=r'closest example ($\beta\to\infty$)'),
     Line2D([], [], color='0.3', ls=':', lw=0.8, label=r'majority-rule prototypes $\mathrm{sgn}\,\bar\eta^\mu$'),
     __import__('matplotlib.patches', fromlist=['Patch']).Patch(facecolor='0.5', edgecolor='0.3', hatch='//', alpha=0.3, label='window (18), theory'),
     Line2D([], [], marker='*', ms=8, color='0.5', mec='k', mew=0.5, lw=0, label='best accuracy, data')]
ax.legend(handles=h, fontsize=7.5, handlelength=1.8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.28), columnspacing=1.6)
ax.set_xscale('log'); ax.set_xlabel(r"$\beta$"); ax.set_ylabel("test accuracy, one step"); ax.set_ylim(0.5, 0.95)
ax.set_title("(c) test accuracy, $M=%d$" % M, fontsize=8.5, loc='left')
fig.savefig('paper/figs/fig_mnist.pdf')
