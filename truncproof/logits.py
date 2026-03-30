import lark
from lark.grammar import Terminal, NonTerminal, Symbol
from lark import Lark
import interegular
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set, Union
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
from itertools import combinations
import heapq
from dataclasses import dataclass, field
import time
from pathlib import Path
from transformers import LogitsProcessor
import torch
import math
import os
from tqdm import tqdm
import pickle
import hashlib

from .parser import Parser, RuleMap, Eos
from .lexer import Lexer, consume

from logging import getLogger
logger = getLogger(__name__)
DEBUG = os.environ.get("DEBUG", "0") == "1"

class GrammarLogitsProcessor(LogitsProcessor):
    """LogitsProcessor of TruncProof."""
    def __init__(self, max_tokens: int, grammar_text: str, whitespace_key: str, tokenizer, eos_token_id: Union[None, int, List[int]] = None, cut: int = 0, debug: bool = False):
        super().__init__()
        self.constraint = GrammarConstraint(grammar_text, whitespace_key, tokenizer, eos_token_id)
        self.max_tokens = max_tokens
        self.cut = cut
        self.debug = debug

    def __call__(self, batched_input_ids: torch.LongTensor, batched_scores: torch.FloatTensor) -> torch.FloatTensor:
        assert len(batched_input_ids.shape) == 2, "input must be (Batch, Seqlength)"
        batched_genstr = self.constraint.tokenizer.batch_decode(batched_input_ids[:, self.cut:], skip_special_tokens=True)
        max_new_tokens = self.max_tokens - batched_input_ids.size(1)
        new_score = batched_scores.clone()
        for b, (genstr, scores) in enumerate(zip(batched_genstr, batched_scores)):
            _vocab_mask, cost = self.constraint.get_vocab_mask_from_text(genstr)
            cost = torch.tensor(cost).to(scores.device)
            new_score[b, :] = torch.where(cost <= max_new_tokens, scores, -math.inf)

            if self.debug:
                topk = torch.topk(scores, 5).indices
                constr_topk = torch.topk(new_score[b, :], 5).indices.view(-1,1)
                print(f"iter {batched_input_ids.size(1)}:",
                        (genstr,),
                        [(tid.item(), tok) for tid, tok in zip(
                            topk,
                            self.constraint.tokenizer.convert_ids_to_tokens(topk)
                        )],
                        "->",
                        [(tid.item(), tok, cost[tid.item()].item()) for tid, tok in zip(
                            constr_topk,
                            self.constraint.tokenizer.convert_ids_to_tokens(constr_topk)
                        )],
                )
        return new_score


class GrammarConstraint:
    def __init__(self, grammar_text: str, whitespace_key: str, tokenizer, eos_token_id: Union[None, int, List[int]]):
        self.tokenizer = tokenizer
        self.parser = Parser(grammar_text)
        self.lexer = Lexer(grammar_text, whitespace_key, tokenizer, self.parser.follow_set)
        self.process_whitespace = whitespace_key != ""
        if eos_token_id is None:
            self.eos_token_id = tokenizer.eos_token_id
        elif type(eos_token_id) is int:
            self.eos_token_id = [eos_token_id]
        else:
            self.eos_token_id = eos_token_id

        self.optimum_expansion_costs: Dict[NonTerminal, int] = {}
        """Shortest length of llm-tokens expandable from each nonterminal"""
        hashnum = hashlib.sha256(b"\0".join(map(lambda x: x.encode("utf-8"), [grammar_text, whitespace_key, tokenizer.__class__.__name__]))).hexdigest()
        cachekeys = ["optimum_expansion_costs"]
        cachefile = Path(f"constr_{hashnum}.pickle")
        if cachefile.exists():
            logger.info(f"loading cache from {str(cachefile)} ...")
            with cachefile.open("rb") as f:
                cache = pickle.load(f)
                for key in cachekeys:
                    setattr(self, key, cache["self"][key])
        else:
            for nonterm in tqdm(self.parser.nonterminals):
                self.optimum_expansion_costs[nonterm] = self.get_optimum_expansion_cost(nonterm, tokenizer)
            with cachefile.open("wb") as f:
                pickle.dump({
                    "self": {key: getattr(self, key) for key in cachekeys},
                }, f)

        if DEBUG:
            self.optimum_expansions: Dict[NonTerminal, List[int]] = {}
            """Shortest llm-tokens expandable from each nonterminal"""
            for nonterm in tqdm(self.parser.nonterminals):
                self.optimum_expansions[nonterm] = self.get_optimum_expansion(nonterm, tokenizer)

    def get_optimum_expansion_cost(self, nonterm: NonTerminal, tokenizer) -> int:
        @dataclass(order=True)
        class ExpansionCost:
            cost: int
            left_term: Optional[Terminal] = field(compare=False)
            trailing_symbols: List[Symbol] = field(compare=False)

        rules = self.parser.rules
        q: List[ExpansionCost] = []
        heapq.heappush(q, ExpansionCost(0, None, [nonterm]))
        while True:
            node = heapq.heappop(q)
            if len(node.trailing_symbols) == 0:
                # fully expanded.
                return node.cost
            head_symbol = node.trailing_symbols[0]
            assert not head_symbol.is_term
            for expansion in rules[head_symbol]:
                cost = node.cost
                new_symbols = list(expansion) + node.trailing_symbols[1:]
                new_trailing_symbols = []
                left = node.left_term
                for i, ns in enumerate(new_symbols):
                    if isinstance(ns, Terminal):
                        fsm_name = new_symbols[i].name
                        if self.process_whitespace:
                            if left is None or (left.name, fsm_name) in self.lexer.confusion_termpairs:
                                # must have whitespace
                                fsm_name = f"--{fsm_name}"
                        fsm = self.lexer.fsm[fsm_name]
                        cost += self.lexer.acceptance_length[fsm_name][fsm.initial]
                        left = ns
                    else:
                        new_trailing_symbols = new_symbols[i:]
                        break
                heapq.heappush(q, ExpansionCost(cost, left, new_trailing_symbols))

    def get_optimum_expansion(self, nonterm: NonTerminal, tokenizer, debug=False) -> List[int]:
        # Only for DEBUG
        @dataclass(order=True)
        class ExpansionCost:
            cost: int
            tokens: List[int] = field(compare=False)
            decoded: str = field(compare=False)
            left_term: Optional[Terminal] = field(compare=False)
            trailing_symbols: List[Symbol] = field(compare=False)

        rules = self.parser.rules
        q: List[ExpansionCost] = []
        heapq.heappush(q, ExpansionCost(0, [], "", None, [nonterm]))
        while True:
            node = heapq.heappop(q)
            if debug:
                print(node)
                time.sleep(1)
            if len(node.trailing_symbols) == 0:
                # fully expanded.
                return node.tokens
            head_symbol = node.trailing_symbols[0]
            assert not head_symbol.is_term
            for expansion in rules[head_symbol]:
                cost = node.cost
                tokens = node.tokens.copy()
                new_symbols = list(expansion) + node.trailing_symbols[1:]
                new_trailing_symbols = []
                left = node.left_term
                for i, ns in enumerate(new_symbols):
                    if isinstance(ns, Terminal):
                        fsm_name = new_symbols[i].name
                        if self.process_whitespace:
                            if left is None or (left.name, fsm_name) in self.lexer.confusion_termpairs:
                                # must have whitespace
                                fsm_name = f"--{fsm_name}"
                        fsm = self.lexer.fsm[fsm_name]
                        tokens += self.lexer.acceptance_tokens[fsm_name][fsm.initial]
                        cost += self.lexer.acceptance_length[fsm_name][fsm.initial]
                        left = ns
                    else:
                        new_trailing_symbols = new_symbols[i:]
                        break
                heapq.heappush(q, ExpansionCost(cost, tokens, tokenizer.decode(tokens, skip_special_tokens=True), left, new_trailing_symbols))
    
    def predict_pdcost(self, stack: List[Symbol]):
        """Predict minimum cost to reduce all symbols in the parser stack."""
        cost = 0
        for symbol in stack:
            if symbol.is_term:
                if self.process_whitespace:
                    # always prefix whitespace for easy prediction
                    fsm_name = f"--{symbol.name}"
                else:
                    fsm_name = symbol.name
                fsm = self.lexer.fsm[fsm_name]
                cost += self.lexer.acceptance_length[fsm_name][fsm.initial]
            else:
                cost += self.optimum_expansion_costs[symbol]
        return cost
    
    def get_vocab_mask_from_text(self, text: str):
        lexical_tokens, reminder = self.lexer.partial_lex(text)
        #print(text)
        #print(lexical_tokens, reminder)
        terms = self.lexer.clean_tokens(lexical_tokens)
        pdstack = self.parser.initialize()
        pdstack = self.parser.run(pdstack, terms)

        # Token ids that will be accepted by the grammar.
        vocab_mask = np.zeros(self.lexer.vocab_size, dtype=bool)
        # The predicted token-cost when each token id is selected.
        INF = np.iinfo(np.int32).max
        minimum_cost = np.full(self.lexer.vocab_size, fill_value=INF, dtype=np.int32)

        # Get length-2 acceptance sequence and their future costs.
        acceptance_sequences: List[Tuple[Tuple[Terminal, ...], int]] = []
        term: Terminal|Eos
        for term in self.parser.valid_inputs(pdstack):
            future_pdstack, success = self.parser.run_if_possible(pdstack, [term])
            if not success:
                continue
            if term == Eos():
                # Output EOS immediately
                acceptance_sequences.append(((), 0))
                continue
            for term2 in self.parser.valid_inputs(future_pdstack):
                future2_pdstack, success = self.parser.run_if_possible(future_pdstack, [term2])
                if not success:
                    continue
                if term2 == Eos():
                    acceptance_sequences.append(((term,), 0))
                    continue
                acceptance_sequences.append(((term, term2), self.predict_pdcost(future2_pdstack)))

        for seq, future_cost in acceptance_sequences:
            if len(seq) == 0:
                # output EOS
                vocab_mask[self.eos_token_id] = True
                minimum_cost[self.eos_token_id] = 1
                continue

            fsm_name: str
            fsm_reminder_state: int
            if len(seq) == 1:
                fsm_name = seq[0].name
                fsm = self.lexer.fsm[fsm_name]
                fsm_reminder_state = consume(fsm, fsm.initial, reminder)
                if fsm_reminder_state is None:
                    # lexer will not feed this terminal
                    continue
                if fsm_reminder_state in fsm.finals:
                    # first term is already accepted. we can output EOS immediately
                    vocab_mask[self.eos_token_id] = True
                    minimum_cost[self.eos_token_id] = 1
                    # we can expand current term. do not `continue`.
            else:
                fsm_name = f"{seq[0].name}-{seq[1].name}"
                fsm = self.lexer.fsm[fsm_name]
                fsm_reminder_state = consume(fsm, fsm.initial, reminder)
                if fsm_reminder_state is None:
                    # lexer will not feed this terminal
                    continue
                assert fsm_reminder_state not in fsm.finals
            new_fsm_state = self.lexer.mask_store[fsm_name][fsm_reminder_state, :].todense() - 1
            vocab_mask |= new_fsm_state >= 0
            cost = np.where(
                new_fsm_state >= 0,
                (
                    1  # select this token id
                    + self.lexer.acceptance_length[fsm_name][new_fsm_state]  # cost to finish this term afterward
                    + future_cost
                ),
                INF  # unreachable
            )
            minimum_cost = np.minimum(minimum_cost, cost)
        return vocab_mask, minimum_cost
    
