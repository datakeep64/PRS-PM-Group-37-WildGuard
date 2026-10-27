"""Give every pipe table relative column widths that follow its content.

Pandoc sizes Word table columns from the dashes in each table's separator row; with
`|---|---|` every column gets the same width, so long labels wrap badly. This rewrites
each separator so its dash counts follow the longest cell of each column.

    python prep_tables.py in.md out.md
"""
import re
import sys


def visible(cell):
    return len(re.sub(r"[*_`]|\[([^\]]*)\]\([^)]*\)", r"\1", cell.strip()))


def main(src, dst):
    lines = open(src, encoding="utf-8").read().split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if (line.startswith("|") and i + 1 < len(lines)
                and re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[i + 1].strip())):
            block = [line]
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                block.append(lines[j])
                j += 1
            rows = [[c for c in r.strip().strip("|").split("|")] for r in block]
            ncol = len(rows[0])
            widths = [max(min(visible(r[k]) if k < len(r) else 0, 34), 10) for r in rows
                      for k in range(ncol)]
            widths = [max(widths[k::ncol]) for k in range(ncol)]
            out.append(line)
            out.append("|" + "|".join("-" * w for w in widths) + "|")
            out.extend(block[1:])
            i = j
            continue
        out.append(line)
        i += 1
    open(dst, "w", encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
