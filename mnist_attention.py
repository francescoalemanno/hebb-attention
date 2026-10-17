"""mnist_attention.py — one attention step with keys = binarised MNIST / fashion-MNIST examples.
Tests: (1) softmax condensation vs REM with the empirical logit spread, (2) three regimes in beta
(test accuracy, n_max, overlap with the class prototype), (3) rho_eff from measured weights.
Usage: python3 mnist_attention.py /path/to/data_dir
"""
import sys, os, gzip, json, time
import numpy as np
from scipy.special import logsumexp, erf
from theory import phi_M

DATA = sys.argv[1] if len(sys.argv) > 1 else "data_mnist"
os.makedirs("results", exist_ok=True)


def load(name, split):
    x = np.frombuffer(gzip.open(f"{DATA}/{name}_{split}_images.gz").read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    y = np.frombuffer(gzip.open(f"{DATA}/{name}_{split}_labels.gz").read(), dtype=np.uint8, offset=8)
    return x.astype(np.float32), y.astype(int)


def binarise(xtr, xte):
    """EPL 2023 recipe: +1 if the pixel exceeds its mean over the training set, else -1."""
    mu = xtr.mean(0)
    return np.where(xtr > mu, 1, -1).astype(np.float32), np.where(xte > mu, 1, -1).astype(np.float32)


def run(name, M, betas, n_test=400, seeds=3):
    xtr, ytr = load(name, "train"); xte, yte = load(name, "test")
    ztr, zte = binarise(xtr, xte)
    N = ztr.shape[1]; K = 10
    out = {b: [] for b in betas}
    extra = []
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], M, replace=False) for c in range(K)])
        keys, lab = ztr[idx], ytr[idx]                       # (K*M, N)
        proto = np.stack([np.sign(keys[lab == c].mean(0) + 1e-9) for c in range(K)])  # majority-rule prototypes
        q_idx = rng.choice(len(zte), n_test, replace=False)
        Q, yq = zte[q_idx], yte[q_idx]
        over = Q @ keys.T / N                                 # (n_test, K*M) overlaps query-key
        # empirical statistics of the within-class overlaps (the data's version of r, sqrt(1-r^4))
        same = lab[None, :] == yq[:, None]
        mean_same = np.array([over[i, same[i]].mean() for i in range(n_test)])
        std_same = np.array([over[i, same[i]].std() for i in range(n_test)])
        # reference readouts
        nc_acc = float(np.mean(np.argmax(Q @ proto.T, 1) == yq))                       # nearest prototype
        nn_acc = float(np.mean(lab[np.argmax(over, 1)] == yq))                           # 1-NN (beta -> inf)
        extra.append(dict(nc_acc=nc_acc, nn_acc=nn_acc, mean_same=float(mean_same.mean()),
                          std_same=float(std_same.mean()), seed=s))
        for b in betas:
            logits = b * N * over
            w = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            h = w @ keys
            state = np.sign(h + 1e-9)
            cls_w = np.stack([w[:, lab == c].sum(1) for c in range(K)], 1)
            acc = float(np.mean(np.argmax(cls_w, 1) == yq))
            W = cls_w[np.arange(n_test), yq]
            ww = w * same
            Meff = (ww.sum(1) ** 2) / np.maximum((ww ** 2).sum(1), 1e-300)
            m_proto = np.array([state[i] @ proto[yq[i]] / N for i in range(n_test)])
            n_max = np.array([(state[i] @ keys[same[i]].T / N).max() for i in range(n_test)])
            x_emp = b * N * std_same                                      # empirical logit spread
            out[b].append(dict(acc=acc, W=float(W.mean()), Meff_over_M=float(Meff.mean() / M),
                               m_proto=float(m_proto.mean()), n_max=float(n_max.mean()), x_emp=float(x_emp.mean())))
    rows = []
    for b in betas:
        rr = out[b]
        row = dict(dataset=name, M=M, beta=float(b))
        for k in rr[0]:
            v = np.array([r[k] for r in rr]); row[k] = float(v.mean()); row[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
        rows.append(row)
    ref = {k: float(np.mean([e[k] for e in extra])) for k in extra[0] if k != "seed"}
    return rows, ref


if __name__ == "__main__":
    betas = np.logspace(-3, 0, 25)
    allrows, refs = [], {}
    t0 = time.time()
    for name in ["mnist", "fmnist"]:
        for M in [16, 64, 256]:
            rows, ref = run(name, M, betas)
            allrows += rows; refs[f"{name}_M{M}"] = ref
            best = max(rows, key=lambda r: r["acc"])
            print(f"{name} M={M}: nearest-prototype acc={ref['nc_acc']:.3f}  1-NN acc={ref['nn_acc']:.3f}  "
                  f"best attention acc={best['acc']:.3f} at beta={best['beta']:.3g} (Meff/M={best['Meff_over_M']:.2f})  "
                  f"<m_same>={ref['mean_same']:.3f} std={ref['std_same']:.3f}   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(rows=allrows, refs=refs), open("results/mnist_attention.json", "w"))
