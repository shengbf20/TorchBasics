"""将 PyTorch 官方教程的 notebook 风格 .py 源文件转换为 Jupyter notebook (.ipynb)。

用法:
    python tools/py2ipynb.py basics --out basics_notebook
    python tools/py2ipynb.py basics/quickstart_tutorial.py --out basics_notebook

特性（对齐官方 sphinx-gallery 生成规则）:
    - 文件开头的模块 docstring -> 首个 markdown cell
    - 由连续 '#' 构成的长行（>=10 个）-> cell 分隔符
    - 分隔符后的连续注释行 -> markdown cell；代码区内的注释保留在代码 cell
    - 注入 Colab 提示 cell（官方同款，注释已中文化）
    - RST -> markdown：链接、行内代码、ATX 标题、note/math/figure 指令、字面块
"""
import json
import re
import sys
from pathlib import Path

SEP_MIN = 10  # 连续 '#' 且长度 >= 此值的行视为 cell 分隔符

INJECTED_CELL = [
    "# 有关在 Google Colab 中运行 notebook 的提示，请参阅",
    "# https://docs.pytorch.org/tutorials/beginner/colab",
    "%matplotlib inline",
]

LINK_RE = re.compile(r"`([^`]+) <([^>]+)>`__?")
CODE2_RE = re.compile(r"``([^`]+)``")
REF_RE = re.compile(r":ref:`([^`]+)`")
MATH_RE = re.compile(r":math:`([^`]+)`")
UNDERLINE_RE = re.compile(r"[=\-~^]{3,}")


# ----------------------------------------------------------------------
# cell 切分
# ----------------------------------------------------------------------
def is_sep(line: str) -> bool:
    s = line.strip()
    return len(s) >= SEP_MIN and set(s) == {"#"}


def is_comment(line: str) -> bool:
    return line.startswith("#")


def comment_content(line: str) -> str:
    if line == "#":
        return ""
    if line.startswith("# "):
        return line[2:]
    return line[1:]


def extract_docstring(lines):
    """提取文件开头的模块 docstring，返回 (doc_lines 或 None, 剩余 lines)。"""
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        return None, lines
    first = lines[idx].strip()
    if not (first.startswith('"""') or first.startswith("'''")):
        return None, lines
    delim = first[:3]
    # 单行 docstring
    if first.count(delim) >= 2:
        inner = first[len(delim):-len(delim)]
        return ([inner] if inner else []), lines[idx + 1:]
    doc = []
    idx += 1
    while idx < len(lines) and delim not in lines[idx]:
        doc.append(lines[idx])
        idx += 1
    idx += 1  # 跳过关闭行
    return doc, lines[idx:]


def split_cells(text: str):
    """返回 [(type, lines), ...]，type 为 'markdown' 或 'code'。"""
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()

    cells = []
    doc, rest = extract_docstring(lines)
    if doc is not None:
        cells.append(("markdown", doc))

    # 按分隔符切段
    segments, cur = [], []
    for ln in rest:
        if is_sep(ln):
            segments.append(cur)
            cur = []
        else:
            cur.append(ln)
    segments.append(cur)

    for seg in segments:
        i = 0
        while i < len(seg) and seg[i].strip() == "":
            i += 1
        md = []
        while i < len(seg) and is_comment(seg[i]):
            md.append(comment_content(seg[i]))
            i += 1
        while md and md[0] == "":
            md.pop(0)
        while md and md[-1] == "":
            md.pop()
        if md:
            cells.append(("markdown", md))
        code = seg[i:]
        while code and code[0] == "":
            code.pop(0)
        while code and code[-1] == "":
            code.pop()
        if code:
            cells.append(("code", code))
    return cells


# ----------------------------------------------------------------------
# RST -> markdown
# ----------------------------------------------------------------------
def convert_inline(line: str) -> str:
    line = REF_RE.sub(r"`\1`", line)
    line = MATH_RE.sub(r"$\1$", line)
    line = LINK_RE.sub(lambda m: "[%s](%s)" % (m.group(1), m.group(2)), line)
    line = CODE2_RE.sub(r"`\1`", line)
    return line


def looks_like_title(prev: str) -> bool:
    p = prev.strip()
    if not p:
        return False
    if p[0] in "#>-|*":
        return False
    if UNDERLINE_RE.fullmatch(p):
        return False
    return True


def md_convert(lines):
    """把 RST 注释/docstring 内容转为 markdown。"""
    out = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s == ".." or (s.startswith(".. _") and s.endswith(":")) or s.startswith(".. include::"):
            i += 1
            continue
        if s.startswith(".. toctree::"):
            i += 1
            while i < n and (lines[i].strip() == "" or lines[i].startswith(" ")):
                i += 1
            continue
        if s.startswith(".. note::"):
            out.append("> **注意：**")
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            while i < n and (lines[i].startswith(" ") or lines[i].strip() == ""):
                body = lines[i].strip()
                out.append("> " + convert_inline(body) if body else ">")
                i += 1
            continue
        if s.startswith(".. math::"):
            out.append("$$")
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            while i < n and lines[i].startswith(" "):
                out.append(lines[i].strip())
                i += 1
            out.append("$$")
            continue
        if s.startswith(".. figure::") or s.startswith(".. image::"):
            i += 1
            while i < n and (lines[i].strip() == "" or lines[i].startswith(" ")):
                i += 1
            continue
        if s.startswith(".. "):
            i += 1
            continue
        line = convert_inline(ln)
        if line.rstrip().endswith("::"):
            line = line.rstrip()[:-2].rstrip()
        out.append(line)
        i += 1

    # 标题下划线 -> ATX 标题
    result = []
    for line in out:
        m = UNDERLINE_RE.fullmatch(line.strip()) if line.strip() else None
        if m and result and looks_like_title(result[-1]):
            level = {"=": 1, "-": 2, "~": 3, "^": 4}[line.strip()[0]]
            result[-1] = "#" * level + " " + result[-1].strip()
        else:
            result.append(line)
    return result


# ----------------------------------------------------------------------
# notebook 组装
# ----------------------------------------------------------------------
def make_cell(cell_type: str, lines) -> dict:
    src = [l + "\n" for l in lines]
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": src,
    }


def make_notebook(cells) -> dict:
    nb_cells = []
    nb_cells.append({"cell_type": "code", "execution_count": None,
                     "metadata": {}, "outputs": [], "source": [l + "\n" for l in INJECTED_CELL]})
    for t, lines in cells:
        nb_cells.append(make_cell(t, lines))
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_file(src: Path, dst: Path):
    text = src.read_text(encoding="utf-8")
    cells = split_cells(text)
    converted = []
    for t, lines in cells:
        if t == "markdown":
            lines = md_convert(lines)
        if not "".join(lines).strip():
            continue  # 丢弃空 cell（如被丢弃的 figure 指令块）
        converted.append((t, lines))
    nb = make_notebook(converted)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    types = [c["cell_type"] for c in nb["cells"]]
    print(f"{src.name:32s} -> {dst.name:32s} cells={len(nb['cells'])} types={types}")


def main(argv):
    out_dir = None
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            out_dir = Path(argv[i + 1])
            i += 2
        else:
            positional.append(a)
            i += 1
    if not positional:
        print(__doc__)
        return 1
    srcs = []
    for a in positional:
        p = Path(a)
        if p.is_dir():
            srcs.extend(sorted(p.glob("*.py")))
        else:
            srcs.append(p)
    if out_dir is None:
        out_dir = Path("basics_notebook")
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in srcs:
        convert_file(s, out_dir / (s.stem + ".ipynb"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
