import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24
from argparse import ArgumentParser
from pathlib import Path

fnames = {
    "syncode": {
        "hard": "syncode_hard.csv",
        "beam": "syncode_beam_hard.csv",
        "mcts": "syncode_mcts_hard.csv",
    },
    "proposed": {
        "hard": "proposed_hard.csv",
        "beam": "proposed_beam_hard.csv",
        "mcts": "proposed_mcts_hard.csv",
    },
}

plt.figure(figsize=(20, 7))
plt.subplots_adjust(left=0.10, right=0.85, bottom=0.15, top=0.90)
limits = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=limits)

x0, y0 = [], []
x1, y1 = [], []
x2, y2 = [], []

entries = [[None for _ in range(2 + len(limits))] for _ in range(6)]  # Method, Strategy, e=1.0, ..., e=1.5

for i, coeff in enumerate(limits):
    folder = Path(f"outputs_gemma-2-2b-it_{coeff:.2f}")
    dfs = {
        method: {
            setting: pd.read_csv(folder / fn, index_col=0)
            for setting, fn in val.items()
        } for method, val in fnames.items()
    }

    df = dfs["syncode"]["hard"]
    exact = df.exact_match.sum()
    schema = df.schema.sum()
    syntax = df.syntax.sum()
    x0.append(i-0.3); y0.append(syntax)
    x1.append(i-0.3); y1.append(schema)
    x2.append(i-0.3); y2.append(exact)
    entries[0][0] = "SynCode"; entries[0][1] = "Greedy"; entries[0][2+i] = exact

    df = dfs["syncode"]["beam"]
    exact = df.exact_match.sum()
    schema = df.schema.sum()
    syntax = df.syntax.sum()
    x0.append(i-0.2); y0.append(syntax)
    x1.append(i-0.2); y1.append(schema)
    x2.append(i-0.2); y2.append(exact)
    entries[1][0] = "SynCode"; entries[1][1] = "BS"; entries[1][2+i] = exact

    df = dfs["syncode"]["mcts"]
    exact = df.exact_match.sum()
    schema = df.schema.sum()
    syntax = df.syntax.sum()
    x0.append(i-0.1); y0.append(syntax)
    x1.append(i-0.1); y1.append(schema)
    x2.append(i-0.1); y2.append(exact)
    entries[2][0] = "SynCode"; entries[2][1] = "MCTS"; entries[2][2+i] = exact

    df = dfs["proposed"]["hard"]
    exact = df.exact_match.sum()
    schema = df.schema.sum()
    syntax = df.syntax.sum()
    x0.append(i+0.0); y0.append(syntax)
    x1.append(i+0.0); y1.append(schema)
    x2.append(i+0.0); y2.append(exact)
    entries[3][0] = "TruncProof"; entries[3][1] = "Greedy"; entries[3][2+i] = exact

    df = dfs["proposed"]["beam"]
    exact = df.exact_match.sum()
    schema = df.schema.sum()
    syntax = df.syntax.sum()
    x0.append(i+0.1); y0.append(syntax)
    x1.append(i+0.1); y1.append(schema)
    x2.append(i+0.1); y2.append(exact)
    entries[4][0] = "TruncProof"; entries[4][1] = "BS"; entries[4][2+i] = exact

    df = dfs["proposed"]["mcts"]
    exact = df.exact_match.sum()
    schema = df.schema.sum()
    syntax = df.syntax.sum()
    x0.append(i+0.2); y0.append(syntax)
    x1.append(i+0.2); y1.append(schema)
    x2.append(i+0.2); y2.append(exact)
    entries[5][0] = "TruncProof"; entries[5][1] = "MCTS"; entries[5][2+i] = exact

plt.bar(x0, y0, align="center", width=0.08, label="syntax")
plt.bar(x1, y1, align="center", width=0.08, label="schema")
plt.bar(x2, y2, align="center", width=0.08, label="exact")
plt.ylabel("Accuracy (%)")
plt.xlabel("Token limit ratio")
plt.title("Accuracy of SynCode (Greedy,BS,MCTS) and Proposed (Greedy,BS,MCTS)")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig("limits.png")

df = pd.DataFrame(entries, columns=["Method", "Strategy"] + [f"e={coeff:.2f}" for coeff in limits])
print(df.to_markdown())

