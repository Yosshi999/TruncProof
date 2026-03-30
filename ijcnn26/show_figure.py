"""
The perplexities provided by LLM against JSON-Mode-Eval.
"""
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--folder", type=Path, default="outputs")
args = parser.parse_args()

fnames = {
    "syncode": {
        "normal": "syncode_normal.csv",
        "hard": "syncode_hard.csv",
    },
    "proposed": {
        "normal": "proposed_normal.csv",
        "hard": "proposed_hard.csv",
    },
}

dfs = {
    method: {
        setting: pd.read_csv(args.folder / fn, index_col=0)
        for setting, fn in val.items()
    } for method, val in fnames.items()
}

gtppl = pd.read_csv(args.folder / "ground_truth_ppl.csv", index_col=0)


plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.10, right=0.995, bottom=0.15, top=0.95)
indices = (dfs["syncode"]["normal"].exact_match == 1) & (dfs["syncode"]["hard"].exact_match == 0)
plt.plot(dfs["syncode"]["normal"].loc[indices].perplexity, "b.", label="Exact-matched", markersize=15)
plt.plot(dfs["syncode"]["hard"].loc[indices].perplexity, "rx", label="Reached limit", markersize=9)
plt.plot(gtppl.loc[indices].ppl, "g_", label="Ground Truth", markersize=15)
plt.grid(axis="x")
plt.xlim(-1, 99.9)
plt.ylim(2, 8)
plt.xlabel("Problem No.")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig(args.folder / "ppl_syncode.png")

plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.10, right=0.995, bottom=0.15, top=0.95)
indices = (dfs["proposed"]["normal"].exact_match == 1) & (dfs["proposed"]["hard"].exact_match == 0)
plt.plot(dfs["proposed"]["normal"].loc[indices].perplexity, "b.", label="Exact-matched", markersize=15)
plt.plot(dfs["proposed"]["hard"].loc[indices].perplexity, "rx", label="Reached limit", markersize=9)
plt.plot(gtppl.loc[indices].ppl, "g_", label="Ground Truth", markersize=15)
plt.grid(axis="x")
plt.xlim(-1, 99.9)
plt.ylim(2, 8)
plt.xlabel("Problem No.")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig(args.folder / "ppl_proposed.png")
