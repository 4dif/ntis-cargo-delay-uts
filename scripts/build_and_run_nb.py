"""Bangun PROPOSAL_UTS_KELOMPOK_08.ipynb dari nb_source.py lalu eksekusi sel demi sel, menyimpan output."""
import ast, base64, contextlib, io, json, os, subprocess, sys, time, traceback

SRC, OUT = sys.argv[1], sys.argv[2]
WORKDIR = sys.argv[3] if len(sys.argv) > 3 else "."
text = open(SRC).read().replace("__NTIS_PREP__", open(os.path.join(os.path.dirname(SRC), "ntis_prep.py")).read().rstrip("\n"))

cells, cur, kind = [], [], None
for line in text.splitlines():
    if line.startswith("# %%"):
        if kind is not None:
            cells.append((kind, cur))
        kind, cur = ("markdown" if "[markdown]" in line else "code"), []
    else:
        cur.append(line)
cells.append((kind, cur))


def clean(lines, md):
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if md:
        lines = [l[2:] if l.startswith("# ") else ("" if l == "#" else l) for l in lines]
    return "\n".join(lines)


nb_cells = []
for i, (k, lines) in enumerate(cells):
    src = clean(lines, k == "markdown")
    c = {"cell_type": k, "metadata": {}, "source": src.splitlines(True), "id": f"cell-{i:02d}"}
    if k == "code":
        c.update({"execution_count": None, "outputs": []})
    nb_cells.append(c)

nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}}

if "--no-run" not in sys.argv:
    os.chdir(WORKDIR); sys.path.insert(0, os.getcwd())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = {"__name__": "__main__"}
    figs = []

    def _show(*a, **k):
        for num in plt.get_fignums():
            f = plt.figure(num); b = io.BytesIO(); f.savefig(b, format="png", dpi=80, bbox_inches="tight")
            figs.append(base64.b64encode(b.getvalue()).decode())
        plt.close("all")
    plt.show = _show
    count = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]); count += 1; c["execution_count"] = count
        outs, buf = [], io.StringIO()
        t0 = time.time()
        if src.startswith("%%writefile"):
            fn, body = src.split("\n", 1); open(fn.split()[1], "w").write(body + "\n")
            outs.append({"output_type": "stream", "name": "stdout", "text": [f"Writing {fn.split()[1]}\n"]})
        elif src.startswith("!pip"):
            outs.append({"output_type": "stream", "name": "stdout", "text": ["(instalasi dilewati pada eksekusi lokal)\n"]})
        else:
            tree = ast.parse(src); last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(tree, f"cell{count}", "exec"), G)
                    val = eval(compile(last, f"cell{count}", "eval"), G) if last else None
                    if figs == [] and plt.get_fignums():
                        _show()
            except Exception:
                print(buf.getvalue()); traceback.print_exc(); sys.exit(f"FAILED at cell {count}")
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
            for f in figs:
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": f, "text/plain": ["<Figure>"]}})
            figs.clear()
            if last is not None and val is not None:
                data = {"text/plain": repr(val).splitlines(True)}
                if hasattr(val, "to_html"):
                    data["text/html"] = val.to_html().splitlines(True)
                outs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
        c["outputs"] = outs
        print(f"[cell {count}] ok ({time.time()-t0:.1f}s)", file=sys.stderr)

json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print("notebook written:", OUT, "cells:", len(nb["cells"]), file=sys.stderr)
