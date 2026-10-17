"""Inline formulas of a LaTeX source, their estimated printed width, and which ones to set as displays.

A long inline formula either overflows the margin or forces loose lines; in the SciPost version, which forbids
line breaks inside inline formulas (\binoppenalty=\relpenalty=10000), it overflows. This module estimates the
width of every inline formula in em (one em = the width of an "M" of the text font) from its symbols and, above
a threshold, moves it to its own display line, carrying the punctuation that follows it. Formulas in captions,
footnotes, titles and item labels are left alone.

Usage, from paper/:
    python inline_math.py main.tex                 # writes main_display.tex, the source with the wide formulas as displays
    python inline_math.py main.tex -t 10 -o out.tex
    python inline_math.py main.tex --table         # only the table of all inline formulas, widest first, with the decision
The source file is never modified. scipost/build_scipost.py imports displayify and lift_overfull from here.
"""
import re
import sys

DEFAULT_THRESHOLD_EM = 12.0

# approximate widths in em at text size
CHAR_EM = 0.55           # a letter, digit or ordinary symbol
BIN_EM = 0.55 + 0.3      # binary operator with its surrounding medium spaces
REL_EM = 0.55 + 0.6      # relation with its surrounding thick spaces
THIN_EM = {'\\,': 0.17, '\;': 0.28, '\\:': 0.22, '\\!': 0.0, '\\ ': 0.33, '\\quad': 1.0, '\\qquad': 2.0}
ZERO = {'\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg', '\\rm', '\\mathrm', '\\mathcal', '\\mathbb',
        '\\bm', '\\boldsymbol', '\\mathbf', '\\text', '\\textrm', '\\operatorname', '\\displaystyle',
        '\\textstyle', '\\scriptstyle', '\\nonumber', '\\bar', '\\hat', '\\tilde', '\\vec', '\\overline',
        '\\dot', '\\ddot', '\\widehat', '\\widetilde', '\\setminus'}
RELATIONS = {'=', '<', '>', '\\le', '\\ge', '\\leq', '\\geq', '\\ll', '\\gg', '\\lesssim', '\\gtrsim', '\\simeq',
             '\\sim', '\\approx', '\\equiv', '\\to', '\\rightarrow', '\\leftarrow', '\\mapsto', '\\in', '\\ne',
             '\\neq', '\\propto', '\\subset', '\\supset', '\\colon'}
BINARY = {'+', '-', '\\pm', '\\mp', '\\cdot', '\\times', '\\odot', '\\circ', '\\cup', '\\cap', '\\wedge', '\\vee'}
NAMED = {'\\ln': 2, '\\log': 3, '\\exp': 3, '\\erf': 3, '\\sgn': 4, '\\tanh': 4, '\\cosh': 4, '\\sinh': 4,
         '\\arg': 3, '\\max': 3, '\\min': 3, '\\softmax': 7, '\\lim': 3, '\\det': 3, '\\sup': 3, '\\inf': 3}
BIG = {'\\sum': 1.0, '\\int': 0.8, '\\prod': 1.0}
# macros of main.tex
MACROS = {'\\s': 1, '\\x': 1, '\\e': 1, '\\E': 1}
NARROW = {'(': 0.35, ')': 0.35, '[': 0.3, ']': 0.3, '|': 0.3, '/': 0.4, ',': 0.3, '.': 0.25, "'": 0.2, ';': 0.3,
          ':': 0.3, '!': 0.3, '\\{': 0.4, '\\}': 0.4, '\\langle': 0.4, '\\rangle': 0.4, '\\infty': 0.8,
          '\\dots': 1.0, '\\cdots': 1.0, '\\ldots': 1.0}

TOKEN = re.compile(r'\\[a-zA-Z]+|\\.|[{}^_]|[^\\{}^_]')


def _tokens(s):
    return TOKEN.findall(s)


def _group(tokens, i):
    """tokens[i] == '{': return (inner tokens, index after the matching '}')."""
    depth, j = 0, i
    while j < len(tokens):
        if tokens[j] == '{':
            depth += 1
        elif tokens[j] == '}':
            depth -= 1
            if depth == 0:
                return tokens[i + 1:j], j + 1
        j += 1
    return tokens[i + 1:], len(tokens)


def _arg(tokens, i):
    """the argument starting at tokens[i]: a braced group or a single token."""
    if i < len(tokens) and tokens[i] == '{':
        return _group(tokens, i)
    return tokens[i:i + 1], i + 1


def width_em(formula):
    """Estimated printed width of an inline formula, in em."""
    return _width(_tokens(formula))


def _width(tokens):
    w, i = 0.0, 0
    while i < len(tokens):
        t = tokens[i]
        if t == '{':
            inner, i = _group(tokens, i)
            w += _width(inner)
            continue
        if t == '}':
            i += 1
            continue
        if t in ('^', '_'):
            inner, i = _arg(tokens, i + 1)
            w += 0.7 * _width(inner)
            continue
        if t == '\\frac':
            num, i = _arg(tokens, i + 1)
            den, i = _arg(tokens, i)
            w += 0.85 * max(_width(num), _width(den)) + 0.2
            continue
        if t == '\\sqrt':
            if i + 1 < len(tokens) and tokens[i + 1] == '[':
                while tokens[i] != ']':
                    i += 1
                i += 1
            inner, i = _arg(tokens, i + 1)
            w += 0.9 + _width(inner)
            continue
        if t in ('\\eqref', '\\ref', '\\label', '\\cite'):
            inner, i = _arg(tokens, i + 1)
            w += 1.5 if t != '\\label' else 0.0
            continue
        i += 1
        if t.isspace():
            continue
        if t in THIN_EM:
            w += THIN_EM[t]
        elif t in ZERO:
            pass
        elif t in RELATIONS:
            w += REL_EM
        elif t in BINARY:
            w += BIN_EM
        elif t in NAMED:
            w += 0.5 * NAMED[t]
        elif t in BIG:
            w += BIG[t]
        elif t in MACROS:
            w += CHAR_EM * MACROS[t]
        elif t == '\\Meff':
            w += CHAR_EM + 0.7 * 3 * 0.5
        elif t in NARROW:
            w += NARROW[t]
        elif t.startswith('\\'):
            w += CHAR_EM          # \alpha, \beta, \xi, ...
        else:
            w += CHAR_EM
    return w


# --- where inline formulas may be moved to a display

PROTECTED_MACROS = ('\\caption', '\\footnote', '\\section*', '\\section', '\\subsection', '\\paragraph', '\\item[')
PROTECTED_ENVS = ('equation', 'equation*', 'align', 'align*', 'figure', 'table', 'abstract')


def _protected_ranges(tex):
    ranges = []
    for m in re.finditer(r'\\(caption|footnote|section\*?|subsection|paragraph)\s*\{', tex):
        depth, j = 1, m.end()
        while j < len(tex) and depth:
            depth += {'{': 1, '}': -1}.get(tex[j], 0)
            j += 1
        ranges.append((m.start(), j))
    for m in re.finditer(r'\\item\[', tex):
        j = tex.index(']', m.end())
        ranges.append((m.start(), j))
    for env in PROTECTED_ENVS:
        for m in re.finditer(r'\\begin\{' + re.escape(env) + r'\}', tex):
            e = tex.find('\\end{' + env + '}', m.end())
            ranges.append((m.start(), e if e >= 0 else len(tex)))
    return ranges


def inline_formulas(tex):
    """[(start, end, formula, protected)] for every $...$ of the source."""
    prot = _protected_ranges(tex)
    out = []
    for m in re.finditer(r'(?<!\\)\$(?!\$)((?:[^$\\]|\\.)+?)\$', tex):
        s, e = m.start(), m.end()
        protected = any(a <= s < b for a, b in prot)
        out.append((s, e, m.group(1), protected))
    return out


def displayify(tex, threshold_em=DEFAULT_THRESHOLD_EM, select=None):
    """Move inline formulas to equation* displays, taking the punctuation that follows them along:
    every unprotected formula wider than threshold_em, or those for which select(start, formula) is true."""
    pieces, last = [], 0
    for s, e, f, protected in inline_formulas(tex):
        if protected:
            continue
        if select is None:
            if width_em(f) <= threshold_em:
                continue
        elif not select(s, f):
            continue
        before = tex[last:s].rstrip(' ')
        after = tex[e:]
        punct = ''
        if after and after[0] in ',.;:':
            punct, after_start = after[0], e + 1
            if punct == ':':      # a display should not end with a colon
                punct = ';'
        else:
            after_start = e
        pieces.append(before + '\n\\begin{equation*}\n' + f.strip() + punct + '\n\\end{equation*}\n')
        last = after_start
        while last < len(tex) and tex[last] == ' ':
            last += 1
    pieces.append(tex[last:])
    return ''.join(pieces)


OVERFULL = re.compile(r'Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)')


def lift_overfull(tex, log, min_pt=2.0):
    """For each paragraph that the pdflatex log reports as overfull by more than min_pt, move the widest
    unprotected inline formula of that paragraph to a display. Returns (new tex, lifted formulas)."""
    starts = [0]
    for i, ch in enumerate(tex):
        if ch == '\n':
            starts.append(i + 1)
    chosen = set()
    for m in OVERFULL.finditer(log):
        pt, a, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
        if pt < min_pt:
            continue
        lo = starts[a - 1] if a - 1 < len(starts) else len(tex)
        hi = starts[b] if b < len(starts) else len(tex)
        inside = [(width_em(f), s, f) for s, e, f, p in inline_formulas(tex) if lo <= s < hi and not p]
        if inside:
            chosen.add(max(inside)[1])
    lifted = [f for s, e, f, p in inline_formulas(tex) if s in chosen]
    if not chosen:
        return tex, []
    return displayify(tex, select=lambda s, f: s in chosen), lifted


def table(tex, threshold_em=DEFAULT_THRESHOLD_EM):
    rows = [(width_em(f), f, p) for _, _, f, p in inline_formulas(tex)]
    rows.sort(key=lambda r: -r[0])
    lines = [f'{"em":>6}  {"decision":<9} formula']
    for w, f, p in rows:
        d = 'skip' if p else ('DISPLAY' if w > threshold_em else '')
        lines.append(f'{w:6.1f}  {d:<9} {f[:100]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source', help='LaTeX source, e.g. main.tex')
    ap.add_argument('-t', '--threshold', type=float, default=DEFAULT_THRESHOLD_EM, help='width in em above which an inline formula becomes a display (default %(default)s)')
    ap.add_argument('-o', '--output', help='output file (default: <source>_display.tex next to the source)')
    ap.add_argument('--table', action='store_true', help='print the table of inline formulas and write nothing')
    a = ap.parse_args()
    src_path = Path(a.source)
    src = src_path.read_text(encoding='utf-8')
    if a.table:
        print(table(src, a.threshold))
    else:
        out_path = Path(a.output) if a.output else src_path.with_name(src_path.stem + '_display' + src_path.suffix)
        new = displayify(src, a.threshold)
        out_path.write_text(new, encoding='utf-8')
        lifted = [f for _, _, f, p in inline_formulas(src) if not p and width_em(f) > a.threshold]
        print(f'{out_path}: {len(lifted)} inline formulas wider than {a.threshold:g} em set as displays')
        for f in lifted:
            print(f'  {width_em(f):5.1f} em  {f[:90]}')
