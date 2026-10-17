# Attention over noisy examples

Code, data and manuscript for

> F. Alemanno, *Attention over noisy examples: generalization and memorization in the exponential Hopfield network* (2026).

Hebbian learning from examples (Agliari, Alemanno, Barra, De Marzo, Neural Netw. 2022; Alemanno, Aquaro, Kanter, Barra, Agliari, EPL 2023, [arXiv:2203.01304](https://arxiv.org/abs/2203.01304)) — a Hopfield network that sees only noisy examples of the archetypes it has to retrieve — is transferred to the exponential Hopfield model, whose zero-temperature update is one step of softmax attention with the examples as keys. A closed-form signal-to-noise theory gives the one-step overlap with an unseen archetype, `m = erf(1/sqrt(2 rho_eff))`, with an *attention entropy* `rho_eff` built from two statistics of the softmax: the weight of the correct class `W` and the number of pooled examples `M_eff`. Three regimes follow (crosstalk → prototype → memorization), the load `alpha = K/N` drops out inside a window of inverse temperatures that widens as `sqrt(N)`, and memorization is the freezing transition of a random energy model at `beta sqrt(N(1-r^2)) = sqrt(2 ln M)`. The picture is tested on Rademacher data, on MNIST / fashion-MNIST, and against an attention head trained by gradient descent, without labels, to denoise the examples.

## Layout

| file | content |
|---|---|
| `hebbattn.py` | dataset, the four models (pairwise sup/unsup Hebb, exponential/attention sup/unsup), zero-temperature dynamics, observables (`m`, `n_max`, `W`, `M_eff`, `rho_eff`) |
| `theory.py` | closed-form predictions: class weight `W(beta)`, REM participation ratio `Phi_M(x)`, `rho_eff`, one-step magnetization, window edges |
| `paper_sims.py`, `paper_sims_fig1.py` | Monte Carlo sweeps for Figs. 3–7 (means and standard errors) → `results/paper_*.json` |
| `regimes_fig.py` | Fig. 1: attention weights of one query in the three regimes |
| `phase_map.py` | Fig. 2: regime map and one-step magnetization in the (β, ρ) plane, theory only |
| `trained_attention.py` | Fig. 9: attention head with keys = values and one slot per example, trained by gradient descent, without labels, to denoise the examples (numpy, hand-written backprop), against the Hebbian head → `results/E5_trained_attention.json` |
| `mnist_attention.py`, `mnist_fig.py` | Fig. 8: MNIST / fashion-MNIST experiments (data not included, see below) |
| `paper_figs.py` | all manuscript figures (LaTeX look, theory = lines, Monte Carlo = symbols) → `paper/figs/*.pdf` |
| `results/` | all numerical results as JSON |
| `fonts/` | Latin Modern Roman (OpenType) for the figures, GUST Font License |
| `paper/` | `main.tex` (one-column version for arXiv, Charter font, own style), `refs.bib` (references of both versions, formatted with the SciPost bibliography style as SciPost asks), `figs/`, compiled `main.pdf` |
| `paper/scipost/` | the same manuscript in the SciPost Physics template (generated from `../main.tex` by `build_scipost.py`, references from `../refs.bib`) |
| `paper/arxiv_source.zip` | the arXiv source of `main.tex`: the `.tex` file with the bibliography inlined, and the figures it uses, compiling on its own |
| `paper/inline_math.py` | estimates the printed width of every inline formula of `main.tex` and writes `main_display.tex`, the same source with the formulas wider than 12 em set as displays; `build_scipost.py` uses it, plus a pass on the paragraphs that pdflatex still reports as overfull |
| `paper/build_arxiv.sh` | rebuilds `arxiv_source.zip` from `main.tex` and checks that it compiles on its own in an empty folder |

## Reproduce

```bash
pip install numpy scipy matplotlib
python3 paper_sims.py            # ~10 min on 2 cores
python3 paper_sims_fig1.py       # ~3 min
python3 trained_attention.py     # ~6 min
python3 mnist_attention.py data_mnist   # needs the four MNIST and four fashion-MNIST idx files in data_mnist/, see below
python3 paper_figs.py && python3 regimes_fig.py && python3 phase_map.py && python3 mnist_fig.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
./build_arxiv.sh                 # arXiv package, from paper/
python3 inline_math.py main.tex   # main_display.tex, wide inline formulas as displays (--table lists them)
cd scipost && python3 build_scipost.py && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

MNIST and fashion-MNIST are read from gzipped idx files named `mnist_{train,test}_{images,labels}.gz` and `fmnist_{train,test}_{images,labels}.gz`; they can be obtained from the fashion-MNIST repository (`zalandoresearch/fashion-mnist`, folder `data/fashion`) and from any MNIST mirror (e.g. `fgnt/mnist`).

All simulations are pure numpy; the figures use the Latin Modern fonts in `fonts/`.

## License

Code: MIT. The manuscript text and figures are © the author. The fonts in `fonts/` are distributed under the GUST Font License.
