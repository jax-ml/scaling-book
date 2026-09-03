#!/usr/bin/env python3
"""Build the A4 PDF book from the Markdown chapters.

This wraps the existing Markdown combiner, Pandoc Typst export, and a small
Typst post-processing pass so local builds and CI releases use the same
professional layout.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMBINED_MD = ROOT / "scaling-book-combined.md"
TYPST_FILE = ROOT / "scaling-book.typ"
PDF_FILE = ROOT / "scaling-book.pdf"


TYPOGRAPHY_PREAMBLE = r'''#let ink = rgb("#18222b")
#let muted = rgb("#66717c")
#let accent = rgb("#0d5f73")
#let accent-soft = rgb("#f1f7f8")
#let rule = rgb("#cfd9df")
#let hairline = rgb("#e7ecef")
#let code-bg = rgb("#f7f8f9")
#let code-text = rgb("#26333d")
#let table-bg = rgb("#f8fafb")
#let body-font = ("Libertinus Serif", "Linux Libertine", "Charter", "New Computer Modern", "Georgia", "Nimbus Roman", "Times New Roman", "serif")
#let display-font = ("Libertinus Serif", "Linux Libertine", "Charter", "New Computer Modern", "Georgia", "Nimbus Roman", "Times New Roman", "serif")
#let sans-font = ("Roboto", "Liberation Sans", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "sans-serif")
#let mono-font = ("DejaVu Sans Mono", "Menlo", "Consolas", "Courier New", "Courier", "monospace")

#set document(title: "How to Scale Your Model: A Systems View of LLMs on TPUs")
#set text(font: body-font, size: 10.4pt, fill: ink, lang: "en", hyphenate: true)
#set par(justify: true, leading: 0.62em, spacing: 0.78em)
#set list(indent: 1.15em, body-indent: 0.58em, spacing: 0.48em)
#set enum(indent: 1.15em, body-indent: 0.58em, spacing: 0.48em)
#set quote(block: true)
#set footnote(numbering: "1")
#set figure(gap: 0.55em)

#let main-header = context {
  let logical-page-num = counter(page).get().first()
  let physical-page-num = here().page()
  let headings = query(heading.where(level: 1))
  let is-chapter-start = headings.any(h => h.location().page() == physical-page-num)
  
  if not is-chapter-start {
    let active-headings = headings.filter(h => h.location().page() <= physical-page-num)
    let heading-text = if active-headings.len() > 0 { active-headings.last().body } else []
    
    [
      #set text(font: sans-font, size: 7.2pt, fill: muted)
      #if calc.even(logical-page-num) [
        #logical-page-num #h(1.25em) #smallcaps[How to Scale Your Model]
      ] else [
        #align(right)[#smallcaps[#heading-text] #h(1.25em) #logical-page-num]
      ]
      #v(3.5pt)
      #line(length: 100%, stroke: 0.22pt + hairline)
    ]
  }
}

#let main-footer = context {
  let physical-page-num = here().page()
  let headings = query(heading.where(level: 1))
  let is-chapter-start = headings.any(h => h.location().page() == physical-page-num)
  
  if is-chapter-start {
    let logical-page-num = counter(page).get().first()
    align(center)[
      #set text(font: sans-font, size: 7.4pt, fill: muted)
      #logical-page-num
    ]
  }
}

#let horizontalrule = {
  pagebreak(weak: true)
  v(0.7em)
}

#let ch-num-state = state("ch-num", none)
#let chapter-number(num) = ch-num-state.update(num)

#let animated(path, alt: none, width: auto, height: auto) = link("https://jax-ml.github.io/scaling-book/" + path)[
  #image(path, alt: alt, width: width, height: height)
]

#show link: set text(fill: accent)
#show emph: it => text(style: "italic")[#it.body]
#show strong: it => text(weight: "semibold")[#it.body]
#show outline.entry.where(level: 1): it => {
  v(9pt, weak: true)
  text(size: 10.2pt, weight: "semibold")[#it]
}
#show outline.entry.where(level: 2): it => {
  set text(size: 9.1pt, fill: muted)
  h(1.5em)
  it
}
#show footnote.entry: it => {
  set text(size: 7.4pt)
  set par(leading: 0.38em, spacing: 0.12em)
  it
}
#show figure: set block(above: 1.35em, below: 1.35em)
#show figure.caption: it => {
  set text(font: sans-font, size: 7.8pt, fill: muted)
  set par(justify: true, leading: 0.42em, spacing: 0.2em)
  it
}
#show image: it => align(center)[#it]

#show table: it => block(
  width: 100%,
  stroke: (bottom: 0.8pt + ink),
  it
)
#set table(
  stroke: (x, y) => if y == 0 {
    (top: 0.8pt + ink, bottom: 0.45pt + ink)
  } else {
    (bottom: 0.18pt + hairline)
  },
  fill: (x, y) => if y == 0 { none } else if calc.odd(y) { none } else { table-bg },
  inset: (x: 6.5pt, y: 5.5pt),
)
#show table: set text(size: 7.45pt)
#show table: set block(above: 1.35em, below: 1.35em)

#show quote.where(block: true): it => block(
  width: 100%,
  fill: none,
  stroke: (left: 1.6pt + accent),
  inset: (left: 12pt, right: 6pt, top: 2pt, bottom: 2pt),
  breakable: true,
)[
  #set text(size: 9.35pt, fill: rgb("#263641"))
  #set par(justify: true, leading: 0.5em, spacing: 0.25em)
  #it.body
]
#show quote.where(block: true): set block(above: 1.25em, below: 1.25em)

#show raw.where(block: false): it => box(
  fill: code-bg,
  stroke: 0.18pt + hairline,
  radius: 1.8pt,
  inset: (x: 2.2pt, y: 1.05pt),
  outset: (y: 0.35pt),
)[
  #text(font: mono-font, size: 8.05pt, fill: code-text)[#it]
]
#show raw.where(block: true): it => block(
  width: 100%,
  fill: code-bg,
  stroke: (left: 1pt + rule),
  inset: (x: 9pt, y: 7pt),
  breakable: true,
)[
  #set text(font: mono-font, size: 7.9pt)
  #set par(leading: 0.42em)
  #it
]
#show raw.where(block: true): set block(above: 1.25em, below: 1.25em)
#show math.equation.where(block: true): set block(above: 1.25em, below: 1.25em)

#show heading.where(level: 1): it => {
  context {
    let num = ch-num-state.get()
    block(width: 100%, inset: (top: 3.2cm), below: 2.25cm, breakable: false)[
      #set text(font: display-font)
      #if num != none [
        #text(font: sans-font, size: 8.8pt, weight: "semibold", fill: accent)[#smallcaps[Chapter #num]]
        #v(0.55em)
      ]
      #text(size: 26pt, weight: "regular", fill: ink)[#it.body]
      #v(0.8em)
      #line(length: 32mm, stroke: 0.75pt + accent)
    ]
  }
  ch-num-state.update(none)
}

#show heading.where(level: 2): it => block(above: 1.8em, below: 0.72em, breakable: false)[
  #text(font: display-font, size: 14pt, weight: "regular", fill: accent)[#it.body]
]

#show heading.where(level: 3): it => block(above: 1.25em, below: 0.45em, breakable: false)[
  #text(font: sans-font, size: 9.3pt, weight: "semibold", fill: ink)[#smallcaps[#it.body]]
]

#set page(
  paper: "a4",
  fill: white,
  margin: (top: 32mm, bottom: 32mm, left: 32mm, right: 32mm),
  header: none,
  footer: none,
)

#v(1fr)
#block(width: 86%)[
  #set par(justify: false)
  #line(length: 26mm, stroke: 1.1pt + accent)
  #v(8mm)
  #text(font: display-font, size: 38pt, weight: "regular", fill: ink)[How to Scale]
  #v(1mm)
  #text(font: display-font, size: 38pt, weight: "regular", fill: ink)[Your Model]
  #v(6mm)
  #text(font: sans-font, size: 14.5pt, fill: accent)[A Systems View of LLMs on TPUs]
  #v(18mm)
  #text(font: body-font, size: 11pt, fill: muted)[A comprehensive guide to training and serving large language models]
  #v(24mm)
  #text(font: sans-font, size: 8pt, fill: ink)[
    Jacob Austin · Sholto Douglas · Roy Frostig · Anselm Levskaya · Charlie Chen · Sharad Vikram \
    Federico Lebron · Peter Choy · Vinay Ramasesh · Albert Webson · Reiner Pope
  ]
]
#v(1fr)

#pagebreak()
#v(8mm)
#text(font: display-font, size: 24pt, fill: ink)[Contents]
#v(4mm)
#outline(title: none, depth: 2)

#pagebreak()
#counter(page).update(1)
#set page(
  paper: "a4",
  fill: white,
  margin: (top: 27mm, bottom: 29mm, left: 34mm, right: 34mm),
  header: main-header,
  footer: main-footer,
)
'''


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def make_tables_auto(typst_content: str) -> str:
    # Matches columns: (31.11%, 5.93%, ...)
    def repl(match):
        cols_str = match.group(1)
        cols = [c.strip() for c in cols_str.split(',') if c.strip()]
        auto_cols = ", ".join(["auto"] * len(cols))
        return f"columns: ({auto_cols},)"
    return re.sub(r'columns:\s*\(([^)]+)\)', repl, typst_content, flags=re.DOTALL)


def get_available_fonts() -> set[str]:
    try:
        res = subprocess.run(["typst", "fonts"], capture_output=True, text=True, check=True)
        return {line.strip() for line in res.stdout.splitlines() if line.strip()}
    except Exception as e:
        print(f"Warning: Failed to run 'typst fonts': {e}", file=sys.stderr)
        return set()


def get_font_definition(name: str, candidates: list[str], available: set[str], default_fallback: list[str]) -> str:
    if not available:
        quoted = ", ".join(f'"{c}"' for c in default_fallback)
        return f'#let {name} = ({quoted})'
    
    filtered = []
    lower_available = {f.lower(): f for f in available}
    for c in candidates:
        if c in available:
            filtered.append(c)
        elif c.lower() in lower_available:
            filtered.append(lower_available[c.lower()])
            
    if not filtered:
        for f in default_fallback:
            if f in available:
                filtered.append(f)
            elif f.lower() in lower_available:
                filtered.append(lower_available[f.lower()])
        if not filtered:
            filtered = default_fallback
            
    quoted = ", ".join(f'"{f}"' for f in filtered)
    return f'#let {name} = ({quoted})'


def get_image_size(filepath: Path) -> tuple[int, int] | None:
    import struct
    try:
        with open(filepath, 'rb') as f:
            data = f.read(100)
            if data.startswith(b'\x89PNG\r\n\x1a\n'):
                # PNG
                w, h = struct.unpack('>II', data[16:24])
                return w, h
            elif data.startswith(b'GIF89a') or data.startswith(b'GIF87a'):
                # GIF
                w, h = struct.unpack('<HH', data[6:10])
                return w, h
            elif data.startswith(b'\xff\xd8'):
                # JPEG
                f.seek(0)
                f.read(2) # skip SOI
                while True:
                    marker, length = struct.unpack('>HH', f.read(4))
                    if marker in (0xffc0, 0xffc2): # SOF0 or SOF2
                        h, w = struct.unpack('>HH', f.read(5)[1:5])
                        return w, h
                    else:
                        f.seek(length - 2, 1)
    except Exception:
        pass
    return None


def apply_typst_layout() -> None:
    typst = TYPST_FILE.read_text()
    intro = "= Introduction\n<introduction>"
    intro_index = typst.find(intro)
    if intro_index == -1:
        raise RuntimeError("Could not find the Introduction heading in Pandoc Typst output")

    # Get available fonts and dynamically filter the lists to prevent compilation warnings
    available = get_available_fonts()
    body_decl = get_font_definition(
        "body-font",
        ["Libertinus Serif", "Linux Libertine", "Charter", "New Computer Modern", "Georgia", "Nimbus Roman", "Times New Roman"],
        available,
        ["Libertinus Serif", "Linux Libertine", "Charter", "New Computer Modern", "Georgia", "Nimbus Roman", "Times New Roman"]
    )
    display_decl = get_font_definition(
        "display-font",
        ["Libertinus Serif", "Linux Libertine", "Charter", "New Computer Modern", "Georgia", "Nimbus Roman", "Times New Roman"],
        available,
        ["Libertinus Serif", "Linux Libertine", "Charter", "New Computer Modern", "Georgia", "Nimbus Roman", "Times New Roman"]
    )
    sans_decl = get_font_definition(
        "sans-font",
        ["Roboto", "Liberation Sans", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial"],
        available,
        ["Roboto", "Liberation Sans", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial"]
    )
    mono_decl = get_font_definition(
        "mono-font",
        ["DejaVu Sans Mono", "Menlo", "Consolas", "Courier New", "Courier"],
        available,
        ["DejaVu Sans Mono", "Menlo", "Consolas", "Courier New", "Courier"]
    )

    preamble = TYPOGRAPHY_PREAMBLE
    preamble = re.sub(r'#let body-font = \([^\)]+\)', body_decl, preamble)
    preamble = re.sub(r'#let display-font = \([^\)]+\)', display_decl, preamble)
    preamble = re.sub(r'#let sans-font = \([^\)]+\)', sans_decl, preamble)
    preamble = re.sub(r'#let mono-font = \([^\)]+\)', mono_decl, preamble)

    typst = preamble + typst[intro_index:]
    
    # 1. Convert GIF image references to animated references
    typst = re.sub(r'image\("([^"\n]+\.gif)"', r'animated("\1"', typst)
    
    # 2. Resize images/animations dynamically based on aspect ratio
    def resize_images(typst_content: str) -> str:
        def img_repl(match):
            func_name = match.group(1) # image or animated
            img_path = match.group(2) # assets/img/...
            rest = match.group(3) or ""
            
            full_path = ROOT / img_path
            size = get_image_size(full_path)
            
            width_str = ""
            height_str = ""
            if size:
                w, h = size
                ar = w / h
                if w < 300:
                    width_str = "width: 180pt"
                elif ar > 2.0:
                    width_str = "width: 95%"
                elif ar > 1.3:
                    width_str = "width: 85%"
                elif ar > 0.8:
                    width_str = "width: 65%"
                else:
                    height_str = "height: 7.5cm"
            else:
                width_str = "width: 75%"
                
            dims = width_str if width_str else height_str
            
            # Remove any existing width or height arguments from clean_rest to prevent duplication
            clean_rest = rest.strip()
            clean_rest = re.sub(r',\s*(width|height):\s*[^,)]+', '', clean_rest)
            
            if clean_rest:
                if clean_rest.startswith(','):
                    return f'{func_name}("{img_path}"{clean_rest.rstrip(",")}, {dims})'
                else:
                    return f'{func_name}("{img_path}", {dims}, {clean_rest})'
            else:
                return f'{func_name}("{img_path}", {dims})'
        
        return re.sub(r'\b(image|animated)\("([^"]+)"([^)]*)\)', img_repl, typst_content)

    typst = resize_images(typst)
    
    # 3. Post-process headings, figures, tables, etc.
    typst = re.sub(r"#strong\[Chapter ([0-9]+)\]", r'#chapter-number("\1")', typst)
    typst = typst.replace("times.circle", "times.o")
    # Clean up Figure prefix and capitalize the first letter of the caption
    typst = re.sub(
        r"(caption:\s*\[\s*)Figure:\s*([A-Za-z])",
        lambda m: f"{m.group(1)}{m.group(2).upper()}",
        typst
    )
    # Also capitalize if it starts with Figure: but doesn't have letters immediately after (fallback)
    typst = re.sub(r"(caption:\s*\[\s*)Figure:\s*", r"\1", typst)
    
    # Auto-size table columns to prevent squished text
    typst = make_tables_auto(typst)
    
    TYPST_FILE.write_text(typst)


def main() -> int:
    run([sys.executable, "bin/convert_to_single_md.py"])
    run(["pandoc", str(COMBINED_MD.name), "-o", str(TYPST_FILE.name)])
    apply_typst_layout()
    run(["typst", "c", str(TYPST_FILE.name), str(PDF_FILE.name)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
