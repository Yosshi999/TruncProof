import xgrammar as xgr
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken
from outlines.models.transformers import TransformerTokenizer
from outlines.processors import CFGLogitsProcessor, GuideLogitsProcessor
from outlines.processors.guide import CFGGuide
from syncode.parsers.grammars import Grammar
from syncode import SyncodeLogitsProcessor
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
import math

from datasets import load_dataset

from truncproof import GrammarLogitsProcessor
from truncproof.generation_utils import mct_search

class OutlinesWrapper(CFGLogitsProcessor):
    def __init__(self, grammar, tokenizer, backend, num_beams=1, eos_token_id=None):
        super().__init__(grammar, tokenizer, backend)
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        if type(eos_token_id) is int:
            self.eos_token_id = [eos_token_id]
        else:
            self.eos_token_id = eos_token_id
        self.num_beams = num_beams

    def iter_valid_token_ids(self, state, candidate_token_ids):
        """Copied from outlines/processors/guide.py and support multiple EOS"""
        for token_id in candidate_token_ids:
            if token_id in self.eos_token_id:
                if self.guide.can_terminate_state(state):
                    yield token_id
            else:
                try:
                    self.guide._get_parser_state_token_applied(state, int(token_id))
                    yield token_id
                except (
                    ValueError,
                    EOFError,
                    UnexpectedToken,
                    UnexpectedCharacters,
                    DedentError,
                ):
                    pass

    def __call__(self, input_ids, logits):
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0]) # type: ignore

        sequence_states: List = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids: # type: ignore
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(self.tensor_adapter.to_list(gen_ids)))
            if curr_state_key not in self._guide_states: # pragma: no cover
                prev_state = self._guide_states[hash(tuple(self.tensor_adapter.to_list(gen_ids[:-1])))]
                curr_state = self.guide.get_next_state(prev_state, self.tensor_adapter.to_scalar(gen_ids[-1]))
                self._guide_states[curr_state_key] = curr_state
            sequence_states.append(self._guide_states[curr_state_key])
        mask = self.tensor_adapter.full_like(logits, -math.inf)
        for i, guide_state in enumerate(sequence_states):
            j = 0
            for legal_token in self.iter_valid_token_ids(guide_state, self.tensor_adapter.argsort_descending(logits[i])):
                mask[i, [legal_token]] = logits[i, [legal_token]] # type: ignore
                j += 1
                if j >= self.num_beams:
                    break
        return mask

class SyncodeWrapper(SyncodeLogitsProcessor):
    """Syncode does not support beamsearch because of its behavior of incremental parser.
    So use this processor for beamsearch. It disables incremental process.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for idx in range(len(input_ids)):
            self.grammar_engine.inc_parser.reset()
            scores[idx:idx+1] = super().__call__(input_ids[idx:idx+1], scores[idx:idx+1])
        return scores

class XGrammarWrapper(xgr.contrib.hf.LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seq_start_idx = None
    def __call__(self, input_ids, scores):
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])
        # re-initialize matchers for each iteration
        self.batch_size = input_ids.shape[0]
        self.compiled_grammars = (
            self.compiled_grammars
            if len(self.compiled_grammars) > 1
            else self.compiled_grammars * self.batch_size
        )
        assert (
            len(self.compiled_grammars) == self.batch_size
        ), "The number of compiled grammars must be equal to the batch size."
        self.matchers = [
            xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)
        ]
        self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)
        
        for i in range(self.batch_size):
            gen_ids = input_ids[i, self._seq_start_idx:]
            for j in range(len(gen_ids)):
                sampled_token = gen_ids[j]
                assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        # We only support masking logits on CUDA or CPU
        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)
        return scores


GENPROMPT = False
parser = ArgumentParser()
parser.add_argument("model", choices=["gemma", "llama"])
parser.add_argument("--hard_only", action="store_true")
parser.add_argument("--normal_only", action="store_true")
parser.add_argument("--limit", type=float, default=1.1, help="coeff of token limit in hard settings")
parser.add_argument("--method", nargs="*", default=["original", "syncode", "outlines", "xgrammar", "proposed"])
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
OUTPUT = Path(f"outputs_{repo_stem}_{limit_coeff:.2f}")
OUTPUT.mkdir(exist_ok=True)

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

text_grammar_for_outlines = r"""
// Based on RFC 8259
?start: value

?value: object
| array
| STRING
| NUMBER
| "true"             -> true
| "false"            -> false
| "null"             -> null

object: "{" [member ("," member)*] "}"
member: STRING ":" value
array : "[" [value ("," value)*] "]"

NUMBER: MINUS? INT FRAC? EXP?
MINUS: "-"
INT: "0" | ("1".."9") DIGIT*
DIGIT: "0".."9"
FRAC: "." DIGIT+
EXP: ("e"|"E") ["+"|"-"] DIGIT+

STRING: /"([^"\\\x00-\x19]|\\["\\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"/
WS: /[ \t\f\r\n]/+
%ignore WS
"""

model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
model = model.to("cuda")

ds = load_dataset("NousResearch/json-mode-eval", split = "train")
problems = [{
    "prompt": [{
        'content':
            problem['prompt'][0]['content']
            + problem['prompt'][1]['content']
            + "\nOnly output JSON.\n",
        'role': 'user'
    }],
    "schema": problem["schema"],
    "ground_truth": problem["completion"],
    "gt_tokens": len(tokenizer.encode(problem["completion"], add_special_tokens=False)),
} for problem in ds]

#problems = problems[:3]

tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = grammar_compiler.compile_builtin_json_grammar()

outlines_tokenizer = TransformerTokenizer(tokenizer)

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

def predict_hard_mcts(kwargs_getter):
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
        tic = time.perf_counter()
        with torch.no_grad():
            output_ids = mct_search(
                model,
                input_ids,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer = tokenizer,
                **kwargs_getter(input_ids, max_new_tokens),
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
    log = SyncodeWrapper(Grammar(text_grammar), tokenizer)
    log.grammar_engine.reset()
    log.grammar_engine.start_from = input_ids.size(1)
    return {
        "logits_processor": [log],
        "num_beams": 10,
    }

def get_syncode_kwargs_mcts(input_ids, max_new_tokens: int):
    log = SyncodeWrapper(Grammar(text_grammar), tokenizer)
    log.grammar_engine.reset()
    log.grammar_engine.start_from = input_ids.size(1)
    return {
        "logits_processor": log,
        "n_trials": 20,
        "puct_coeff": 5,
        "max_new_tokens": max_new_tokens
    }

def get_outlines_kwargs(input_ids, max_new_tokens: int):
    logi = OutlinesWrapper(text_grammar_for_outlines, outlines_tokenizer, "torch", eos_token_id=model.config.eos_token_id)
    return {"logits_processor": [logi]}

def get_outlines_kwargs_beam(input_ids, max_new_tokens: int):
    logi = OutlinesWrapper(text_grammar_for_outlines, outlines_tokenizer, "torch", num_beams=10, eos_token_id=model.config.eos_token_id)
    return {
        "logits_processor": [logi],
        "num_beams": 10,
    }

def get_outlines_kwargs_mcts(input_ids, max_new_tokens: int):
    logi = OutlinesWrapper(text_grammar_for_outlines, outlines_tokenizer, "torch", num_beams=10, eos_token_id=model.config.eos_token_id)
    return {
        "logits_processor": logi,
        "n_trials": 20,
        "puct_coeff": 5,
        "max_new_tokens": max_new_tokens
    }

def get_xgrammar_kwargs(input_ids, max_new_tokens: int):
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    return {"logits_processor": [xgr_logits_processor]}

def get_xgrammar_kwargs_beam(input_ids, max_new_tokens: int):
    xgr_logits_processor = XGrammarWrapper(compiled_grammar)
    return {
        "logits_processor": [xgr_logits_processor],
        "num_beams": 10,
    }

def get_xgrammar_kwargs_mcts(input_ids, max_new_tokens: int):
    xgr_logits_processor = XGrammarWrapper(compiled_grammar)
    return {
        "logits_processor": xgr_logits_processor,
        "n_trials": 20,
        "puct_coeff": 5,
        "max_new_tokens": max_new_tokens
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

def get_proposed_kwargs_mcts(input_ids, max_new_tokens: int):
    logiproc = GrammarLogitsProcessor(input_ids.size(1) + max_new_tokens, text_grammar, "", tokenizer, model.config.eos_token_id, input_ids.size(1), debug=False)
    return {
        "logits_processor": logiproc,
        "n_trials": 20,
        "puct_coeff": 5,
        "max_new_tokens": max_new_tokens
    }


methods_normal = {
    "original": (lambda _a, _b: {}),
    "syncode": get_syncode_kwargs,
    "outlines": get_outlines_kwargs,
    "xgrammar": get_xgrammar_kwargs,
    "proposed": get_proposed_kwargs,
}

methods_hard = {
    "original": (lambda _a, _b: {}),
    "original_beam": (lambda _a, _b: {"num_beams": 10}),
    "syncode": get_syncode_kwargs,
    "syncode_beam": get_syncode_kwargs_beam,
    "outlines": get_outlines_kwargs,
    "outlines_beam": get_outlines_kwargs_beam,
    "xgrammar": get_xgrammar_kwargs,
    "xgrammar_beam": get_xgrammar_kwargs_beam,
    "proposed": get_proposed_kwargs,
    "proposed_beam": get_proposed_kwargs_beam,
}

methods_hard_mcts = {
    "syncode_mcts": get_syncode_kwargs_mcts,
    "xgrammar_mcts": get_xgrammar_kwargs_mcts,
    "outlines_mcts": get_outlines_kwargs_mcts,
    "proposed_mcts": get_proposed_kwargs_mcts,
}

# ==== groundtruth ====
ppls = []
for problem in tqdm(problems):
    input_ids = torch.LongTensor(
        tokenizer.apply_chat_template(problem["prompt"], tokenize=True, add_generation_prompt=GENPROMPT)
    ).to(model.device).unsqueeze(0)  # (batchsize, sequence)
    gt = tokenizer.encode(problem["ground_truth"], add_special_tokens=False, return_tensors="pt").to(model.device)
    output_ids = torch.cat([input_ids, gt], dim=1)
    ppls.append(ppl(output_ids))
pd.DataFrame({"ppl": ppls}).to_csv(OUTPUT / "ground_truth_ppl.csv")

# ==== Nmax: 400 ====
if not args.hard_only:
    print("Normal setting")
    for key, kwargs_getter in methods_normal.items():
        if key.split("_")[0] not in args.method:
            continue
        print(key)
        with (OUTPUT / f"{key}_normal.pickle").open("wb") as f:
            completions, predicted_tokens, perplexity, gentime = predict(kwargs_getter)
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

# ==== Hard setting ====
if not args.normal_only:
    print("Hard setting e=", limit_coeff)
    for key, kwargs_getter in methods_hard.items():
        if key.split("_")[0] not in args.method:
            continue
        with (OUTPUT / f"{key}_hard.pickle").open("wb") as f:
            completions, predicted_tokens, max_new_tokens, perplexity, gentime = predict_hard(kwargs_getter)
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

# ==== Hard (MCTS) setting ====
if not args.normal_only:
    print("Hard (MCTS) setting e=", limit_coeff)
    for key, kwargs_getter in methods_hard_mcts.items():
        if key.split("_")[0] not in args.method:
            continue
        with (OUTPUT / f"{key}_hard.pickle").open("wb") as f:
            completions, predicted_tokens, max_new_tokens, perplexity, gentime = predict_hard_mcts(kwargs_getter)
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

