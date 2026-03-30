import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--folder", type=Path, default="outputs")
args = parser.parse_args()

fnames = {
    "base": {
        #"normal": "original_normal.csv",
        "hard": "original_hard.csv",
    },
    "syncode": {
        #"normal": "syncode_normal.csv",
        "hard": "syncode_hard.csv",
        "beam": "syncode_beam_hard.csv",
        "mcts": "syncode_mcts_hard.csv",
    },
    "xgrammar": {
        #"normal": "xgrammar_normal.csv",
        "hard": "xgrammar_hard.csv",
        "beam": "xgrammar_beam_hard.csv",
        "mcts": "xgrammar_mcts_hard.csv",
    },
    "outlines": {
        #"normal": "outlines_normal.csv",
        "hard": "outlines_hard.csv",
        "beam": "outlines_beam_hard.csv",
        #"mcts": "outlines_mcts_hard.csv",
    },
    "proposed": {
        #"normal": "proposed_normal.csv",
        "hard": "proposed_hard.csv",
        "beam": "proposed_beam_hard.csv",
        "mcts": "proposed_mcts_hard.csv",
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

#table_normal = pd.DataFrame([
#    {"Method": "No constraint", **scores["base"]["normal"]},
#    {"Method": "Outlines", **scores["outlines"]["normal"]},
#    {"Method": "SynCode",       **scores["syncode"]["normal"]},
#    {"Method": "XGrammar", **scores["xgrammar"]["normal"]},
#    {"Method": "Proposed",      **scores["proposed"]["normal"]},
#])
#print("### Normal ###")
#print(table_normal)
#print()
#
table_hard = pd.DataFrame([
    {"Method": "No constraint", **scores["base"]["hard"], "tps": tps["base"]["hard"]},

    {"Method": "Outlines", **scores["outlines"]["hard"], "tps": tps["outlines"]["hard"]},
    {"Method": "Outlines+BS", **scores["outlines"]["beam"], "tps": tps["outlines"]["beam"]},
    #{"Method": "Outlines+MCTS", **scores["outlines"]["mcts"], "tps": tps["outlines"]["mcts"]},

    {"Method": "SynCode",       **scores["syncode"]["hard"], "tps": tps["syncode"]["hard"]},
    {"Method": "SynCode+BS",    **scores["syncode"]["beam"], "tps": tps["syncode"]["beam"]},
    {"Method": "SynCode+MCTS",    **scores["syncode"]["mcts"], "tps": tps["syncode"]["mcts"]},

    {"Method": "XGrammar", **scores["xgrammar"]["hard"], "tps": tps["xgrammar"]["hard"]},
    {"Method": "XGrammar+BS", **scores["xgrammar"]["beam"], "tps": tps["xgrammar"]["beam"]},
    {"Method": "XGrammar+MCTS", **scores["xgrammar"]["mcts"], "tps": tps["xgrammar"]["mcts"]},

    {"Method": "Proposed",      **scores["proposed"]["hard"], "tps": tps["proposed"]["hard"]},
    {"Method": "Proposed+BS",   **scores["proposed"]["beam"], "tps": tps["proposed"]["beam"]},
    {"Method": "Proposed+MCTS", **scores["proposed"]["mcts"], "tps": tps["proposed"]["mcts"]},
])
pd.options.display.float_format = '{:.2f}'.format
print("### Hard ###")
print(table_hard)
print()
