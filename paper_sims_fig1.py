"""paper_sims_fig1.py — larger-N data for Fig. 4 (the load disappears) (three well-separated plateaus).
(a) N=4000, beta=0.012, rho in {0.06,0.25,0.5}, one step only;  (b) N=4000, alpha=0.2, rho scan, one step + fixed point.
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
from hebbattn import trial, r_of

N, M, BETA = 4000, 16, 0.012
SEEDS, NQ = 3, 6


def run(args):
    p, s, i = args
    rng = np.random.default_rng(777 + 100 * s + i)
    K = max(2, int(round(p["alpha"] * N)))
    return trial(N, K, M, r_of(p["rho"], M), p["model"], p["lam"], rng, n_queries=NQ, T=p["T"])


def sweep(grid, name):
    t0 = time.time()
    jobs = [(p, s, i) for i, p in enumerate(grid) for s in range(SEEDS)]
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=1)
    rows = []
    for i, p in enumerate(grid):
        rr = res[i * SEEDS:(i + 1) * SEEDS]
        row = dict(p)
        for k in rr[0]:
            v = np.array([x[k] for x in rr], dtype=float)
            row[k] = float(v.mean()); row[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
        rows.append(row)
    json.dump(rows, open(f"results/{name}.json", "w"))
    print(f"[{name}] {len(grid)} points in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    alphas = [0.01, 0.02, 0.035, 0.06, 0.1, 0.16, 0.25, 0.35, 0.5]
    grid = [dict(alpha=a, rho=rh, model=m, lam=l, N=N, M=M, T=1)
            for (m, l) in [("pairwise_sup", None), ("attn_unsup", BETA)] for rh in [0.06, 0.25, 0.5] for a in alphas]
    sweep(grid, "paper_S1a_N4000")
    rhos = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]
    grid = [dict(alpha=0.2, rho=rh, model=m, lam=l, N=N, M=M, T=10)
            for (m, l) in [("pairwise_sup", None), ("attn_unsup", BETA)] for rh in rhos]
    sweep(grid, "paper_S1b_N4000")
