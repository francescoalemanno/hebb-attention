"""Generate the SciPost Physics version of the manuscript from ../main.tex (figures from ../figs, references from ../refs.bib).

Usage: python build_scipost.py [threshold_em]  (run from paper/scipost/)
"""
import re
import sys
from pathlib import Path

import subprocess

sys.path.insert(0, "..")
from inline_math import displayify, lift_overfull  # noqa: E402

THRESHOLD_EM = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0   # inline formulas wider than this go to a display

src = Path("../main.tex").read_text()
lines = src.splitlines(keepends=True)


def between(start_pat, end_pat):
    s = next(i for i, l in enumerate(lines) if l.startswith(start_pat))
    e = next(i for i, l in enumerate(lines) if i > s and l.startswith(end_pat))
    return "".join(lines[s:e])


abstract = between(r"\begin{abstract}", r"\end{abstract}")
abstract = abstract.replace(r"\begin{abstract}", "").replace(r"\noindent", "").strip()
# same plain-text abstract as main.tex, so that arXiv and SciPost coincide
abstract = r"""In Hebbian learning from examples, a Hopfield network retrieves archetypes seen only through noisy examples. We study it in the exponential Hopfield model: one softmax attention step with the unlabelled examples as keys. A signal-to-noise analysis gives the overlap with an unseen archetype in closed form: a majority-rule law in which an attention entropy replaces the dataset entropy. At high temperature the archetype is lost in the crosstalk; at low temperature the softmax freezes on the closest training example, as in a random energy model, and the layer memorizes. In between, in a window widening with the square root of the network size, the layer pools the examples of the correct class and retrieves the unseen archetype as well as their majority vote allows, at any fixed load. On structured datasets, such as MNIST and fashion-MNIST, the best accuracy lies beyond the freezing."""
body = between(r"\section{Introduction}", r"\section*{Acknowledgments}")
appendix = between(r"\section{Details of the simulations}", r"\end{document}")

# single column: wide figures keep the text width, column figures shrink
body = body.replace(r"\begin{figure*}[t]", r"\begin{figure}[t]").replace(r"\end{figure*}", r"\end{figure}")
body = body.replace(r"width=\columnwidth", r"width=0.62\textwidth")
# equations split for the narrow column fit on one line here
body = body.replace(r"\Big]\nonumber\\&\quad+\tau^2(1-q^2),\nonumber", r"\Big]+\tau^2(1-q^2),\nonumber")
# the combined Derrida 1985 / Derrida-Gardner 1986 entry becomes two BibTeX entries
body = body.replace(r"\cite{Derrida1985}", r"\cite{Derrida1985,DerridaGardner1986}")
# inline formulas too wide for the SciPost line (\binoppenalty=\relpenalty=10000 forbid breaking them) are set
# as displays, by the width estimate of ../inline_math.py; "python ../inline_math.py main.tex --table" lists them
body = displayify(body, THRESHOLD_EM)
appendix = displayify(appendix, THRESHOLD_EM)
appendix = appendix.replace(r"\section{Small-$x$ expansion of $\Phi_M$}", r"\section{Small-\texorpdfstring{$x$}{x} expansion of \texorpdfstring{$\Phi_M$}{Phi\_M}}")

preamble = r"""\documentclass{SciPost}

\graphicspath{{../}}
\binoppenalty=10000
\relpenalty=10000
\setlength{\emergencystretch}{1em}

\hypersetup{
    colorlinks,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}

\usepackage[bitstream-charter]{mathdesign}
\usepackage{amsthm,bm,orcidlink}
\urlstyle{same}

\DeclareSymbolFont{usualmathcal}{OMS}{cmsy}{m}{n}
\DeclareSymbolFontAlphabet{\mathcal}{usualmathcal}

\fancypagestyle{SPstyle}{
\fancyhf{}
\lhead{\colorbox{scipostblue}{\bf \color{white} ~SciPost Physics }}
\rhead{{\bf \color{scipostdeepblue} ~Submission }}
\renewcommand{\headrulewidth}{1pt}
\fancyfoot[C]{\textbf{\thepage}}
}

\theoremstyle{plain}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{lemma}{Lemma}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}

\newcommand{\s}{\bm{\sigma}}
\newcommand{\x}{\bm{\xi}}
\newcommand{\e}{\bm{\eta}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Meff}{M_{\rm eff}}
\newcommand{\erf}{\mathrm{erf}}
\newcommand{\sgn}{\mathrm{sign}}
\newcommand{\softmax}{\mathrm{softmax}}

\begin{document}

\pagestyle{SPstyle}

\begin{center}{\Large \textbf{\color{scipostdeepblue}{
Attention over noisy examples: generalization and memorization\\ in the exponential Hopfield network
}}}\end{center}

\begin{center}\textbf{
Francesco Alemanno\,\orcidlink{0000-0003-1065-2590}\textsuperscript{1$\star$}
}\end{center}

\begin{center}
{\bf 1} Independent researcher, Italy
\\[\baselineskip]
$\star$ \href{mailto:francescoalemanno710@gmail.com}{\small francescoalemanno710@gmail.com}
\end{center}


\section*{\color{scipostdeepblue}{Abstract}}
\textbf{\boldmath{%
""" + abstract + r"""
}}

\vspace{\baselineskip}

%%%%%%%%%% BLOCK: Copyright information
\noindent\textcolor{white!90!black}{%
\fbox{\parbox{0.975\linewidth}{%
\textcolor{white!40!black}{\begin{tabular}{lr}%
  \begin{minipage}{0.6\textwidth}%
    {\small Copyright attribution to authors. \newline
    This work is a submission to SciPost Physics. \newline
    License information to appear upon publication. \newline
    Publication information to appear upon publication.}
  \end{minipage} & \begin{minipage}{0.4\textwidth}
    {\small Received Date \newline Accepted Date \newline Published Date}%
  \end{minipage}
\end{tabular}}
}}
}
%%%%%%%%%% BLOCK: Copyright information

\linenumbers

\vspace{2pt}
\noindent\rule{\textwidth}{1pt}
{\setlength{\cftbeforesecskip}{5pt}\tableofcontents}
\noindent\rule{\textwidth}{1pt}
\vspace{10pt}

"""

ack = r"""\section*{Acknowledgements}
The research on this topic was accelerated by the use of Claude Fable 5.1 (Anthropic); the author takes full responsibility for the content. Code, data and the Monte Carlo results reproducing all figures are available at \url{https://github.com/francescoalemanno/hebb-attention}.

\paragraph{Funding information}
This work was supported by SolarSud T.R.E. di Miccoli Antonia, Italy (\url{https://solarsud.it}).

\begin{appendix}
\numberwithin{equation}{section}

"""

end = r"""\end{appendix}

\bibliography{../refs}

\end{document}
"""

out = preamble + body + ack + appendix + end
Path("main.tex").write_text(out)
print("wrote main.tex:", out.count("\n"), "lines")

# second pass: paragraphs that pdflatex still reports as overfull (several medium formulas on one line) lose their
# widest inline formula to a display; the copyright block of the template is reported too but has no formula to lift
for _ in range(6):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], stdout=subprocess.DEVNULL, check=True)
    tex = Path("main.tex").read_text()
    new, lifted = lift_overfull(tex, Path("main.log").read_text(encoding="latin-1"))
    if not lifted:
        break
    for f in lifted:
        print("display (overfull paragraph):", f[:80])
    Path("main.tex").write_text(new)
for f in ("main.aux", "main.log", "main.out", "main.toc"):
    Path(f).unlink(missing_ok=True)
