import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--folder", type=Path, default="outputs")
args = parser.parse_args()

fnames = {
    "xgrammar": {
        "normal": "xgrammar_normal.csv",
        "hard": "xgrammar_hard.csv",
        "beam": "xgrammar_beam_hard.csv",
    },
    "outlines": {
        "normal": "outlines_normal.csv",
        "hard": "outlines_hard.csv",
        "beam": "outlines_beam_hard.csv",
    },
}

dfs = {
    method: {
        setting: pd.read_csv(args.folder / fn, index_col=0)
        for setting, fn in val.items()
    } for method, val in fnames.items()
}

scores = {
    method: {
        setting: d[["syntax", "schema", "exact_match"]].sum().to_dict()
        for setting, d in val.items()
    } for method, val in dfs.items()
}
print(scores)

tps = {
    method: {
        setting: d.predicted_tokens.sum() / d.generation_time.sum()
        for setting, d in val.items()
    } for method, val in dfs.items()
}

table_normal = pd.DataFrame([
    {"Method": "XGrammar", **scores["xgrammar"]["normal"]},
    {"Method": "Outlines", **scores["outlines"]["normal"]},
])
print("### Normal ###")
print(table_normal)
print()

table_hard = pd.DataFrame([
    {"Method": "XGrammar", **scores["xgrammar"]["hard"], "tps": tps["xgrammar"]["hard"]},
    {"Method": "XGrammar+BS", **scores["xgrammar"]["beam"], "tps": tps["xgrammar"]["beam"]},
    {"Method": "Outlines", **scores["outlines"]["hard"], "tps": tps["outlines"]["hard"]},
    {"Method": "Outlines+BS", **scores["outlines"]["beam"], "tps": tps["outlines"]["beam"]},
])
pd.options.display.float_format = '{:.2f}'.format
print("### Hard ###")
print(table_hard)
print()

