from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import torch
from typing import List
from tqdm import tqdm
import time
import pickle
from pathlib import Path
import json
from dataclasses import dataclass
from jsonschema import validate, ValidationError
import pandas as pd
from argparse import ArgumentParser

from datasets import load_dataset

from syncode.parsers.grammars import Grammar
from syncode import SyncodeLogitsProcessor
from truncproof import GrammarLogitsProcessor
from truncproof.generation_utils import mct_search

class FallbackSyncodeLogitsProcessor(SyncodeLogitsProcessor):
    """Syncode does not support beamsearch because of its behavior of incremental parser.
    So use this processor for beamsearch. It disables incremental process.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for idx in range(len(input_ids)):
            self.grammar_engine.inc_parser.reset()
            scores[idx:idx+1] = super().__call__(input_ids[idx:idx+1], scores[idx:idx+1])
        return scores

GENPROMPT = False
parser = ArgumentParser()
parser.add_argument("model", choices=["gemma", "llama"])
parser.add_argument("--hard_only", action="store_true")
parser.add_argument("--normal_only", action="store_true")
parser.add_argument("--original", action="store_true", help="use original prompt")
parser.add_argument("--old_grammar", action="store_true", help="use the grammar provided by SynCode")
parser.add_argument("--limit", type=float, default=1.1, help="coeff of token limit in hard settings")
args = parser.parse_args()

repo_map = {
    "gemma": "google/gemma-2-2b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
}
tokenizer_options = {
    "gemma": {},
    "llama": dict(use_fast=False, legacy=False),
}

repo = repo_map[args.model]
tokenizer = AutoTokenizer.from_pretrained(repo, **tokenizer_options[args.model])

repo_stem = repo.split("/")[-1]
limit_coeff = args.limit
OUTPUT = Path(f"outputs_{repo_stem}_{limit_coeff:.2f}{'_original' if args.original else ''}{'_oldgrammar' if args.old_grammar else ''}_compact")
OUTPUT.mkdir()

if args.old_grammar:
    skip_term = "WS"
    text_grammar = r"""
// Adapted from https://github.com/lapp0/outlines
?start: start_value

?start_value: object
| array

?value: object
| array
| EMPTY_STRING
| NONEMPTY_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : NONEMPTY_STRING ":" value

NONEMPTY_STRING: /\"[^"]+\"/
EMPTY_STRING: /\"\"/

DIGIT: "0".."9"
HEXDIGIT: "a".."f"|"A".."F"|DIGIT
INT: DIGIT+
SIGNED_INT: ["+"|"-"] INT
DECIMAL: INT "." INT? | "." INT


_EXP: ("e"|"E") SIGNED_INT
FLOAT: INT _EXP | DECIMAL _EXP?
NUMBER: FLOAT | INT
SIGNED_NUMBER: ["+"|"-"] NUMBER
WS: /[ \t\f\r\n]/+

%ignore WS
"""
else:
    skip_term = ""
    text_grammar = r"""
// Based on RFC 8259
?start: value

_BEGIN_ARRAY:     /[ \t\f\r\n]*\[[ \t\f\r\n]*/
_BEGIN_OBJECT:    /[ \t\f\r\n]*\{[ \t\f\r\n]*/
_END_ARRAY:       /[ \t\f\r\n]*\][ \t\f\r\n]*/
_END_OBJECT:      /[ \t\f\r\n]*\}[ \t\f\r\n]*/
_NAME_SEPARATOR:  /[ \t\f\r\n]*:[ \t\f\r\n]*/
_VALUE_SEPARATOR: /[ \t\f\r\n]*,[ \t\f\r\n]*/

?value: object
| array
| STRING
| number
| "true"             -> true
| "false"            -> false
| "null"             -> null

object: _BEGIN_OBJECT [member (_VALUE_SEPARATOR member)*] _END_OBJECT
member: STRING _NAME_SEPARATOR value
array : _BEGIN_ARRAY [value (_VALUE_SEPARATOR value)*] _END_ARRAY

number: MINUS? INT FRAC? EXP?
MINUS: "-"
INT: "0" | ("1".."9") DIGIT*
DIGIT: "0".."9"
FRAC: "." DIGIT+
EXP: ("e"|"E") ["+"|"-"] DIGIT+

STRING: /"([^"\\\x00-\x19]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"/
"""

model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
model = model.to("cuda")

ds = load_dataset("NousResearch/json-mode-eval", split = "train")
problems = [{
    "prompt": [{
        'content':
            problem['prompt'][0]['content']
            + problem['prompt'][1]['content']
            + ("\n" if args.original else "\nOnly output JSON. Eliminate white spaces and keep the output as compact as possible.\n"),
        'role': 'user'
    }],
    "schema": problem["schema"],
    "ground_truth": problem["completion"],
    "gt_tokens": len(tokenizer.encode(problem["completion"], add_special_tokens=False)),
} for problem in ds]

#problems = problems[:3]


print("#### sample input ####")
print(tokenizer.apply_chat_template(problems[0]["prompt"], tokenize=False))
print("#### sample ground truth ####")
print(problems[0]["ground_truth"])
print()


@dataclass
class EvalResult:
    syntax: int = 0
    schema: int = 0
    exact_match: int = 0


def validate_one(problem: dict, pred: str) -> EvalResult:
    schema = json.loads(problem["schema"])
    try:
        parsed = json.loads(pred)
    except json.decoder.JSONDecodeError:
        return EvalResult()  # syntax error
    if not isinstance(parsed, dict):
        return EvalResult(1)  # schema error
    try:
        validate(instance=parsed, schema=schema)
    except ValidationError as e:
        #print(e)
        return EvalResult(1)  # schema error
    if parsed == json.loads(problem["ground_truth"]):
        return EvalResult(1, 1, 1)  # exact match
    else:
        return EvalResult(1, 1, 0)  # schema is correct but unmatch

@torch.no_grad()
def ppl(output_ids) -> float:
    logits = model(output_ids, return_dict=True).logits[:, :-1, :].log_softmax(-1)
    selected_logits = logits.take_along_dim(output_ids[:, 1:].unsqueeze(-1), dim=-1).squeeze(-1)
    return torch.exp(-selected_logits.mean()).item()

def predict(kwargs_getter):
    completions = []
    predicted_tokens = []
    perplexity = []
    generation_time = []
    for problem in tqdm(problems):
        input_ids = torch.LongTensor(
            tokenizer.apply_chat_template(problem["prompt"], tokenize=True, add_generation_prompt=GENPROMPT)
        ).to(model.device).unsqueeze(0)  # (batchsize, sequence)
        input_length = input_ids.size(1)
        kwargs = kwargs_getter(input_ids, 400)
        tic = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=400,
                do_sample=False,
                **kwargs,
            )
        output = tokenizer.decode(output_ids.tolist()[0][input_length:], skip_special_tokens=True)
        tac = time.perf_counter()
        completions.append(output)
        predicted_tokens.append(output_ids.size(1) - input_length)
        perplexity.append(ppl(output_ids))
        generation_time.append(tac - tic)
    return completions, predicted_tokens, perplexity, generation_time

def predict_hard(kwargs_getter):
    completions = []
    predicted_tokens = []
    max_new_tokens_array = []
    perplexity = []
    generation_time = []
    for problem in tqdm(problems):
        input_ids = torch.LongTensor(
            tokenizer.apply_chat_template(problem["prompt"], tokenize=True, add_generation_prompt=GENPROMPT)
        ).to(model.device).unsqueeze(0)  # (batchsize, sequence)
        input_length = input_ids.size(1)
        gt_length = problem["gt_tokens"]
        max_new_tokens = int(gt_length * limit_coeff)
        kwargs = kwargs_getter(input_ids, max_new_tokens)
        tic = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
        output = tokenizer.decode(output_ids.tolist()[0][input_length:], skip_special_tokens=True)
        tac = time.perf_counter()
        completions.append(output)
        predicted_tokens.append(output_ids.size(1) - input_length)
        max_new_tokens_array.append(max_new_tokens)
        perplexity.append(ppl(output_ids))
        generation_time.append(tac - tic)
    return completions, predicted_tokens, max_new_tokens_array, perplexity, generation_time


def get_syncode_kwargs(input_ids, max_new_tokens: int):
    log = SyncodeLogitsProcessor(Grammar(text_grammar), tokenizer)
    log.grammar_engine.reset()
    log.grammar_engine.start_from = input_ids.size(1)
    return {"logits_processor": [log]}

def get_syncode_kwargs_beam(input_ids, max_new_tokens: int):
    log = FallbackSyncodeLogitsProcessor(Grammar(text_grammar), tokenizer)
    log.grammar_engine.reset()
    log.grammar_engine.start_from = input_ids.size(1)
    return {
        "logits_processor": [log],
        "num_beams": 10,
    }

def get_proposed_kwargs(input_ids, max_new_tokens: int):
    logiproc = GrammarLogitsProcessor(input_ids.size(1) + max_new_tokens, text_grammar, skip_term, tokenizer, model.config.eos_token_id, input_ids.size(1))
    return {
        "logits_processor": [logiproc],
    }

def get_proposed_kwargs_beam(input_ids, max_new_tokens: int):
    logiproc = GrammarLogitsProcessor(input_ids.size(1) + max_new_tokens, text_grammar, skip_term, tokenizer, model.config.eos_token_id, input_ids.size(1), debug=False)
    return {
        "logits_processor": [logiproc],
        "num_beams": 10,
    }


methods = {
    "original": [(lambda _a, _b: {}), (lambda _a, _b: {})],
    "original_beam": [None, (lambda _a, _b: {"num_beams": 10})],
    "syncode": [get_syncode_kwargs, get_syncode_kwargs],
    "syncode_beam": [None, get_syncode_kwargs_beam],
    "proposed": [get_proposed_kwargs, get_proposed_kwargs],
    "proposed_beam": [None, get_proposed_kwargs_beam],
}

ppls = []
for problem in tqdm(problems):
    input_ids = torch.LongTensor(
        tokenizer.apply_chat_template(problem["prompt"], tokenize=True, add_generation_prompt=GENPROMPT)
    ).to(model.device).unsqueeze(0)  # (batchsize, sequence)
    gt = tokenizer.encode(problem["ground_truth"], add_special_tokens=False, return_tensors="pt").to(model.device)
    output_ids = torch.cat([input_ids, gt], dim=1)
    ppls.append(ppl(output_ids))
pd.DataFrame({"ppl": ppls}).to_csv(OUTPUT / "ground_truth_ppl.csv")

for key, kwargs_getter in methods.items():
    print(key)
    if (not args.hard_only) and kwargs_getter[0] is not None:
        with (OUTPUT / f"{key}_normal.pickle").open("wb") as f:
            completions, predicted_tokens, perplexity, gentime = predict(kwargs_getter[0])
            pickle.dump(completions, f)

        result = []
        for problem, pred in zip(problems, completions):
            result.append(validate_one(problem, pred))
        df = pd.DataFrame(result)
        df["max_new_tokens"] = [400 for _ in problems]
        df["gt_tokens"] = [problem["gt_tokens"] for problem in problems]
        df["predicted_tokens"] = predicted_tokens
        df["prompt"] = [tokenizer.apply_chat_template(problem["prompt"], tokenize=False) for problem in problems]
        df["schema_text"] = [problem["schema"] for problem in problems]
        df["ground_truth"] = [problem["ground_truth"] for problem in problems]
        df["prediction"] = completions
        df["perplexity"] = perplexity
        df["generation_time"] = gentime
        df.to_csv(OUTPUT / f"{key}_normal.csv")

    if (not args.normal_only) and kwargs_getter[1] is not None:
        with (OUTPUT / f"{key}_hard.pickle").open("wb") as f:
            completions, predicted_tokens, max_new_tokens, perplexity, gentime = predict_hard(kwargs_getter[1])
            pickle.dump(completions, f)

        result = []
        for problem, pred in zip(problems, completions):
            result.append(validate_one(problem, pred))
        df = pd.DataFrame(result)
        df["max_new_tokens"] = max_new_tokens
        df["gt_tokens"] = [problem["gt_tokens"] for problem in problems]
        df["predicted_tokens"] = predicted_tokens
        df["prompt"] = [tokenizer.apply_chat_template(problem["prompt"], tokenize=False) for problem in problems]
        df["schema_text"] = [problem["schema"] for problem in problems]
        df["ground_truth"] = [problem["ground_truth"] for problem in problems]
        df["prediction"] = completions
        df["perplexity"] = perplexity
        df["generation_time"] = gentime
        df.to_csv(OUTPUT / f"{key}_hard.csv")

