"""Markdown to HTML, rendered at BUILD time.

Deliberately not a client-side renderer. The memo's whole claim is that its
numbers cannot drift from the code that produced them; shipping a JavaScript
markdown parser alongside it would add a second thing that can go wrong on a
page whose entire selling point is that nothing does.

This handles exactly the subset of markdown the memo uses — headings, tables,
blockquotes, rules, bold/italic/code, simple lists — and nothing else. A general
markdown library would be a dependency for no gain.
"""

from __future__ import annotations

import html as _html
import re

__all__ = ["render", "CSS"]

CSS = """<!doctype html><meta charset="utf-8"><title>Coldspend — client memo</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{max-width:760px;margin:0 auto;padding:3rem 1.25rem 5rem;
 font:16px/1.7 "Segoe UI",system-ui,-apple-system,sans-serif;color:#0b0b0b;background:#fcfcfb}
h1{font-size:1.75rem;line-height:1.28;letter-spacing:-.02em;margin-bottom:1.4rem}
h2{margin-top:2.4rem;font-size:1.18rem;border-bottom:1px solid #e1e0d9;padding-bottom:.3rem}
h3{margin-top:1.7rem;font-size:1.02rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:1.1rem 0}
th,td{border-bottom:1px solid #e1e0d9;padding:.45rem .6rem;text-align:left}
th{color:#52514e;font-weight:600}
blockquote{border-left:3px solid #eb6834;margin:1.2rem 0;padding:.7rem 1.1rem;
 background:#fff8ee;border-radius:0 4px 4px 0}
hr{border:0;border-top:1px solid #e1e0d9;margin:2.2rem 0}
code{background:#e1e0d9;padding:.1rem .35rem;border-radius:3px;font-size:.86em}
p.li{margin:.35rem 0 .35rem 1.2rem}
em{color:#52514e}
a{color:#2a78d6}
@media print{body{padding:0;max-width:none}h2{page-break-after:avoid}}
</style>
"""

_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_ORDERED = re.compile(r"^\d+\. ")


def _inline(text: str) -> str:
    text = _LINK.sub(r'<a href="\2">\1</a>', text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return _CODE.sub(r"<code>\1</code>", text)


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(c) <= set("-: ") for c in cells)


def render(md: str, css: str = CSS) -> str:
    out: list[str] = []
    in_table = False

    for line in md.splitlines():
        raw = line.rstrip()

        if raw.startswith("|") and raw.endswith("|"):
            cells = [c.strip() for c in raw[1:-1].split("|")]
            if _is_separator(cells):
                continue
            tag = "td" if in_table else "th"
            if not in_table:
                out.append("<table>")
                in_table = True
            body = "".join(f"<{tag}>{_inline(_html.escape(c))}</{tag}>" for c in cells)
            out.append(f"<tr>{body}</tr>")
            continue

        if in_table:
            out.append("</table>")
            in_table = False

        if not raw:
            continue

        esc = _html.escape(raw)
        if raw.startswith("### "):
            out.append(f"<h3>{_inline(esc[4:])}</h3>")
        elif raw.startswith("## "):
            out.append(f"<h2>{_inline(esc[3:])}</h2>")
        elif raw.startswith("# "):
            out.append(f"<h1>{_inline(esc[2:])}</h1>")
        elif raw.startswith("---"):
            out.append("<hr>")
        elif raw.startswith("> "):
            out.append(f"<blockquote>{_inline(esc[2:])}</blockquote>")
        elif _ORDERED.match(raw):
            out.append(f"<p class='li'>{_inline(esc)}</p>")
        elif raw.startswith("- "):
            out.append(f"<p class='li'>&bull; {_inline(esc[2:])}</p>")
        else:
            out.append(f"<p>{_inline(esc)}</p>")

    if in_table:
        out.append("</table>")

    return css + "\n".join(out) + "\n"
