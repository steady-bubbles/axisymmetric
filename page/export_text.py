"""Write docs/page_text.tex: all prose of the page as one LaTeX file, for editing.

It holds, in page order, the title, the intro, every section with its paragraphs
and displayed equations, the figure captions, the texts on the three asymptotics
cards, the paragraph on related works, the references and the footer. Figure
labels, legends, tooltips and buttons are left out. The file compiles with
pdflatex, and the citations are numbered as on the page.

Edits made in the .tex are carried back by hand into page/page_template.html,
page/prior_work.html and page/references.json, by diffing the edited file
against the committed one.

Usage: python page/export_text.py [built.html]   (after page/build_page.py, whose output
supplies the numbers that the page's script fills in)
"""
import json
import os
import re
import sys
from html.parser import HTMLParser

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):
    print(__doc__)
    sys.exit()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "page"))
from build_page import CITE, citation_order, load_references  # noqa: E402

MATH = re.compile(r"(\\\(.*?\\\)|\\\[.*?\\\])", re.S)
SKIP_TAGS = {"svg", "fieldset", "label", "button", "input", "script", "style"}
SKIP_CLASSES = {"controls", "legend", "nav", "jumps", "facts", "tip", "note", "ok"}
FIGURE_CLASSES = {"chart", "meridian", "cards"}

PREAMBLE = r"""%% Text of the page "Axisymmetric bubbles", for editing.
%%
%% Edit anything here and send the file back; the changes are then carried into
%% the page. Written by page/export_text.py.
%%
%%  * Every paragraph, displayed equation and caption of the page is here, in
%%    page order. Figure labels, legends and buttons are not.
%%  * [Figure text, ...] marks a caption and says where it sits.
%%  * \cite{key} as usual; the keys are those of the bibliography at the end,
%%    numbered alphabetically as on the page. For a new reference add a
%%    \bibitem with authors, title, journal, year and DOI or arXiv number.
%%  * Maths in \( \) and \[ \] is rendered on the page by MathJax, which knows
%%    amsmath but no macros of your own.
%%  * \val{...} is a number the page computes from the data; it can be moved
%%    or dropped, but its value is not taken from this file.
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[nocompress]{cite}
\usepackage[margin=2.5cm]{geometry}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
\newcommand{\pagetitle}[1]{{\LARGE\bfseries #1\par}\medskip}
\newcommand{\val}[1]{#1}
\newenvironment{figtext}[1]{\par\medskip\noindent{\sffamily\footnotesize [Figure text, #1]}\par\nopagebreak\noindent\ignorespaces}{\par\medskip}
\newcommand{\pagefooter}[1]{\par\bigskip\noindent\rule{\linewidth}{0.4pt}\par\noindent #1\par}

\begin{document}
"""

UNICODE = {"\u2013": "--", "\u2014": "---", "\u2026": r"\dots{}", "\u00a0": "~"}


def tex_text(s, keep_cites=True):
    """Escape a text node for LaTeX, leaving \\( \\) and \\[ \\] maths untouched and
    turning {{cite:a,b}} into \\cite{a,b}."""
    out = []
    for i, part in enumerate(MATH.split(s)):
        if i % 2:
            out.append(part)
            continue
        cites = []
        part = CITE.sub(lambda m: cites.append(m.group(1)) or f"\x00{len(cites) - 1}\x00", part)
        part = re.sub(r"([&%#_$])", r"\\\1", part)
        for u, t in UNICODE.items():
            part = part.replace(u, t)
        part = re.sub(r"\x00(\d+)\x00", lambda m: r"\cite{" + cites[int(m.group(1))].replace(" ", "") + "}", part)
        out.append(part)
    return "".join(out)


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []          # finished LaTeX blocks
        self.buf = None        # inline LaTeX of the open block
        self.block = None      # (tag, classes) of the open block
        self.closers = []      # per open tag: text to append at its end tag
        self.skip = 0          # depth inside skipped subtrees
        self.stack = []
        self.section = ""
        self.card = ""
        self.figure_seen = False

    # -- helpers
    def emit(self, s):
        self.out.append(s)

    def where(self, cls):
        if self.card:
            return "prediction above the chart" if "pred" in cls else "below the chart"
        return "below the figure" if self.figure_seen else "above the figure"

    # -- parser callbacks
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = set((a.get("class") or "").split())
        self.stack.append((tag, cls))
        if self.skip or tag in SKIP_TAGS or cls & SKIP_CLASSES:
            if tag not in ("input", "br"):
                self.skip += 1
            if cls & FIGURE_CLASSES or tag == "svg":
                self.figure_seen = True
            return
        if cls & FIGURE_CLASSES:
            self.figure_seen = True
        if tag == "section":
            self.figure_seen = False
            self.card = ""
        if tag == "div" and "card" in cls:
            self.card = ""
        if tag == "div" and "refs-here" in cls:
            self.emit("\x01REFS\x01")
            return
        if tag in ("h1", "h2", "h3", "p", "footer") or (tag == "div" and "math" in cls):
            self.block = (tag, cls)
            self.buf = []
            return
        if self.buf is None:
            return
        if tag in ("b", "strong"):
            self.buf.append(r"\textbf{")
            self.closers.append((tag, "}"))
        elif tag in ("i", "em"):
            self.buf.append(r"\emph{")
            self.closers.append((tag, "}"))
        elif tag == "a":
            self.buf.append(r"\href{" + a["href"].replace("%", r"\%").replace("#", r"\#") + "}{")
            self.closers.append((tag, "}"))
        elif tag == "br":
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()
        if self.skip:
            if tag not in ("input", "br"):
                self.skip -= 1
            return
        if self.closers and self.closers[-1][0] == tag and self.buf is not None and self.block and self.block[0] != tag:
            self.buf.append(self.closers.pop()[1])
            return
        if self.block and tag == self.block[0] and self.buf is not None:
            text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            btag, cls = self.block
            self.block, self.buf = None, None
            if btag == "div":
                self.emit(text)
            elif btag == "h1":
                self.emit(r"\pagetitle{" + text + "}")
            elif btag == "h2":
                self.section = text
                self.emit(r"\section*{" + text + "}")
            elif btag == "h3":
                self.card = text
                self.emit(r"\subsection*{" + text + "}")
            elif btag == "footer":
                self.emit(r"\pagefooter{" + text + "}")
            elif "caption" in cls or "pred" in cls or "res" in cls:
                self.emit(r"\begin{figtext}{" + self.where(cls) + "}\n" + text + "\n\\end{figtext}")
            else:
                self.emit(("% intro\n" if "lede" in cls else "") + text)

    def handle_data(self, data):
        if self.skip or self.buf is None:
            return
        self.buf.append(data if self.block and self.block[0] == "div" else tex_text(data))


def card_texts(tpl, D):
    """The result lines the page's script writes under the three cards."""
    values = {
        "M.chi_coef[0].toFixed(13)": f"{D['mns']['chi_coef'][0]:.13f}",
        "S.touching.We.toFixed(10)": f"{D['special']['touching']['We']:.10f}",
    }

    def fill(m):
        if m.group(1) not in values:
            raise KeyError(f"no value for ${{{m.group(1)}}} in a card text")
        return r"\val{" + values[m.group(1)] + "}"

    out = {}
    for m in re.finditer(r'\$\("(res-\w+)"\)\.innerHTML = `(.*?)`;', tpl, re.S):
        out[m.group(1)] = re.sub(r"\$\{(.*?)\}", fill, m.group(2).replace("\\\\", "\\"))
    return out


def bibliography(order, refs):
    lines = [r"\begin{thebibliography}{99}"]
    for k in order:
        r = refs[k]
        title = tex_text(r["title"])
        if r.get("url"):
            title = r"\href{" + r["url"].replace("%", r"\%").replace("#", r"\#") + "}{" + title + "}"
        venue = tex_text(r["venue"])
        if r.get("arxiv") and r.get("doi"):
            venue += r", \href{https://arxiv.org/abs/" + r["arxiv"] + "}{arXiv:" + r["arxiv"] + "}"
        elif r.get("arxiv"):
            venue = f"arXiv:{r['arxiv']}" if r["venue"] == "preprint" else f"{venue}, arXiv:{r['arxiv']}"
        lines.append(rf"\bibitem{{{k}}} {tex_text(r['authors'])}, {title}, {venue} ({r['year']}).")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def main():
    tpl = open(os.path.join(ROOT, "page", "page_template.html"), encoding="utf-8").read()
    tpl = tpl.replace("{{prior_work}}", open(os.path.join(ROOT, "page", "prior_work.html"), encoding="utf-8").read())
    built = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "index.html")
    page = open(built, encoding="utf-8").read()
    D, _ = json.JSONDecoder().raw_decode(page, page.index("const D = ") + len("const D = "))
    refs = load_references()
    order = citation_order(tpl, refs)

    body = tpl[tpl.index('<div class="wrap"'):tpl.index("<script")]
    for key, text in card_texts(tpl, D).items():
        empty = f'<p class="res" id="{key}"></p>'
        if empty not in body:
            raise ValueError(f"no placeholder {empty}")
        body = body.replace(empty, f'<p class="res" id="{key}">{text}</p>')
    body = body.replace("{{references}}", '<div class="refs-here"></div>')

    ex = Extract()
    ex.feed(body)
    ex.close()
    tex = PREAMBLE + "\n" + "\n\n".join(ex.out) + "\n\n\\end{document}\n"
    tex = tex.replace("\x01REFS\x01", bibliography(order, refs))
    # the page's references heading is replaced by the bibliography's own
    tex = tex.replace("\\section*{References}\n\n\\begin{thebibliography}", "\\begin{thebibliography}")

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "page_text.tex")
    open(path, "w", encoding="utf-8").write(tex)
    odd = sorted({c for c in tex if ord(c) > 127})
    print(f"wrote {os.path.relpath(path, ROOT)}: {len(ex.out)} blocks, {len(order)} references"
          + (f"; non-ASCII characters: {''.join(odd)}" if odd else ""))


if __name__ == "__main__":
    main()
