"""paper_sims.py — Monte Carlo data for the manuscript figures (means and standard errors over seeds).
Run: python3 paper_sims.py [--quick]      -> results/paper_S*.json
"""
import sys, os, time, json
import numpy as np
from multiprocessing import Pool
from hebbattn import trial, r_of, make_dataset, HebbianMemory, noisy_copy

QUICK = "--quick" in sys.argv
os.makedirs("results", exist_ok=True)
SEEDS = 2 if QUICK else 4
NQ = 3 if QUICK else 8


def _one(args):
    fn, p, s, i = args
    rng = np.random.default_rng(10_000 * s + i)
    return fn(p, rng)


def sweep(fn, grid, name, seeds=SEEDS):
    t0 = time.time()
    jobs = [(fn, p, s, i) for i, p in enumerate(grid) for s in range(seeds)]
    with Pool(2) as pool:
        res = pool.map(_one, jobs, chunksize=4)
    rows = []
    for i, p in enumerate(grid):
        rr = res[i * seeds:(i + 1) * seeds]
        row = dict(p)
        for k in rr[0]:
            v = np.array([x[k] for x in rr], dtype=float)
            row[k] = float(v.mean()); row[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append(row)
    json.dump(rows, open(f"results/{name}.json", "w"))
    print(f"[{name}] {len(grid)} points x {seeds} seeds in {time.time()-t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- S1: (alpha, rho) maps, N=1000
N1, M1 = (600, 16) if QUICK else (1000, 16)
alphas = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5]
rhos = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]
LAM_STAR = 0.025


def run_S1(p, rng):
    K = max(2, int(round(p["alpha"] * N1)))
    r = r_of(p["rho"], M1)
    return trial(N1, K, M1, r, p["model"], p["lam"], rng, n_queries=NQ)


if __name__ == "__main__":
    grid = [dict(alpha=a, rho=rh, model=m, lam=l, N=N1, M=M1)
            for (m, l) in [("pairwise_sup", None), ("attn_unsup", LAM_STAR)] for a in alphas for rh in rhos]
    sweep(run_S1, grid, "paper_S1_maps")

    # ------------------------------------------------------------ S2: lambda scan, N=1000, alpha=0.1
    N2 = 600 if QUICK else 1000
    lams = np.logspace(-2.7, 0, 16 if QUICK else 28)
    cfg2 = [(0.05, 32), (0.2, 16), (0.5, 8)]

    def run_S2(p, rng):
        return trial(p["N"], int(0.1 * p["N"]), p["M"], r_of(p["rho"], p["M"]), "attn_unsup", p["lam"], rng, n_queries=NQ)

    grid = [dict(N=N2, rho=rh, M=m, lam=float(l)) for (rh, m) in cfg2 for l in lams]
    sweep(run_S2, grid, "paper_S2_lambda")

    # ------------------------------------------------------------ S3: N dependence, rho=0.2, M=16
    Ns = [250, 1000] if QUICK else [250, 500, 1000, 2000, 4000]
    grid = [dict(N=n, rho=0.2, M=16, lam=float(l)) for n in Ns for l in lams]
    sweep(run_S2, grid, "paper_S3_N", seeds=min(SEEDS, 3))

    # ------------------------------------------------------------ S4: REM condensation of the softmax
    def run_S4(p, rng):
        out = []
        for q in range(NQ):
            xi, eta = make_dataset(p["N"], 2, p["M"], p["r"], rng)
            mem = HebbianMemory(xi, eta, "attn_unsup", p["lam"])
            s0 = noisy_copy(xi[0], p["r"], rng).astype(np.float32)
            _, w = mem.field(s0)
            ww = w[mem.labels == 0]; ww = ww / ww.sum()
            out.append(1.0 / np.sum(ww ** 2) / p["M"])
        return dict(Meff_over_M=float(np.mean(out)))

    xs = np.logspace(-1, 1.2, 14 if QUICK else 26)
    cfg4 = [(2000, 16, 0.5), (2000, 64, 0.5), (2000, 256, 0.5), (500, 64, 0.5), (2000, 64, 0.8)]
    grid = [dict(N=n, M=m, r=r, x=float(x), lam=float(x / np.sqrt(n * (1 - r ** 4)))) for (n, m, r) in cfg4 for x in xs]
    sweep(run_S4, grid, "paper_S4_rem")
    print("all done")
