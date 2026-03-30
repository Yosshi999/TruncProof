import lark
from lark.grammar import Terminal, NonTerminal, Symbol
from lark.lexer import PatternStr, PatternRE
from lark import Lark
import interegular
import interegular.fsm
import numpy as np
import transformers.utils
from tqdm import tqdm
from scipy.sparse import csr_array

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set, TypeVar, Generic, Generator
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
import pickle
import hashlib
from logging import getLogger
import os
logger = getLogger(__name__)

DEBUG = os.environ.get("DEBUG", "0") == "1"

def sanity_check(tid: int, tokenizer) -> bool:
    return (
        tokenizer.decode([tid]) == tokenizer.decode([tid], skip_special_tokens=True)
    )

class Lexer:
    """Regex based lexer."""
    def __init__(self, text_grammar: str, whitespace_key: str, tokenizer, follow_set):
        self.strict_lex = True
        self.sorted_vocab = []
        self.vocab_size = len(tokenizer.get_vocab())
        """Tokenizer's vocab sorted by token ID. Special token is converted into None."""
        for tid in range(self.vocab_size):
            if sanity_check(tid, tokenizer):
                token = tokenizer.decode([tid], skip_special_tokens=True)
                # sanitize prefix space
                if tokenizer.convert_ids_to_tokens(tid).startswith(transformers.utils.SENTENCEPIECE_UNDERLINE):
                    token = " " + token
                self.sorted_vocab.append(token)
            else:
                self.sorted_vocab.append(None)

        self.fsm: Dict[str, interegular.FSM] = {}
        """FSM that accepts lexical token.
        if key is
          * NAME: FSM accepts its pattern with (possibly) leading whitespaces; /\s*(pattern)/
          * --NAME: FSM accepts its pattern with leading whitespaces; /\s+(pattern)/
          * NAME1-NAME2: FSM accepts /\s*(pattern1)\s*(pattern2)/ , or, if needed, accepts /\s*(pattern1)\s+(pattern2)/
        """

        self.mask_store: Dict[str, csr_array] = {}
        """If mask[s,c] == t >= 1, token-id c leads to a live state t-1 from state s.
        If 0, transition(s,c) fails."""
        #self.mask_store: Dict[str, np.ndarray] = {}
        #"""If mask[s,c] == t >= 0, token-id c leads to a live state t from state s.
        #If -1, transition(s,c) fails."""

        if DEBUG:
            self.acceptance_tokens: Dict[str, List[List[int]]] = {}
            """shortest tokens from each state to accept"""

        self.acceptance_length: Dict[str, np.ndarray] = {}
        """shortest token length from each state to accept"""

        self.confusion_termpairs: Tuple[str, str] = set()
        """To use for confusion check.
        If it has (TERM1, TERM2), <TERM1><TERM2> can be recognized as another term,
        so there must be whitespace between them."""

        parser = Lark(text_grammar, start="start", parser="lalr", lexer="basic", debug=True)
        atom_fsm: Dict[str, interegular.FSM] = {}

        # Bottleneck part. Use cache.
        hashnum = hashlib.sha256(b"\0".join(map(lambda x: x.encode("utf-8"), [text_grammar, whitespace_key, tokenizer.__class__.__name__]))).hexdigest()
        cachefile = Path(f"fsm_{hashnum}.pickle")
        if DEBUG:
            cachekeys = ["fsm", "mask_store", "acceptance_tokens", "acceptance_length", "confusion_termpairs"]
        else:
            cachekeys = ["fsm", "mask_store", "acceptance_length", "confusion_termpairs"]
        if cachefile.exists():
            logger.info(f"loading cache from {str(cachefile)} ...")
            with cachefile.open("rb") as f:
                cache = pickle.load(f)
                for key in cachekeys:
                    setattr(self, key, cache["self"][key])
                # amend anything_else in FSMs
                for fsm in self.fsm.values():
                    for key, value in fsm.alphabet._symbol_mapping.items():
                        if isinstance(key, interegular.fsm.anything_else.__class__):
                            container = fsm.alphabet._symbol_mapping.pop(key)
                            fsm.alphabet._symbol_mapping[interegular.fsm.anything_else] = container
                            break

        else:
            print("constructing FSMs")
            reserved_fsms: List[interegular.FSM] = []
            regex_words: List[str] = []
            for termdef in parser.terminals:
                atom_fsm[termdef.name] = interegular.parse_pattern(termdef.pattern.to_regexp()).to_fsm()
                if isinstance(termdef.pattern, PatternStr):
                    reserved_fsms.append(atom_fsm[termdef.name])
                elif isinstance(termdef.pattern, PatternRE):
                    regex_words.append(termdef.name)
                else:
                    raise NotImplementedError()
            if len(reserved_fsms) > 0:
                # resolve priority
                union_reserved_fsm = reserved_fsms[0].union(*reserved_fsms[1:])
                for term in regex_words:
                    atom_fsm[term] = atom_fsm[term].difference(union_reserved_fsm)  # do not catch reserved words
        
            if whitespace_key == "":
                for termdef in parser.terminals:
                    if termdef.name in parser.ignore_tokens:
                        continue
                    self.fsm[termdef.name] = atom_fsm[termdef.name].reduce()
            else:
                # Define FSM with or without whitespaces
                ws = atom_fsm[whitespace_key]
                for termdef in parser.terminals:
                    if termdef.name in parser.ignore_tokens:
                        continue
                    self.fsm[termdef.name] = ((ws + atom_fsm[termdef.name]) | atom_fsm[termdef.name]).reduce()
                    self.fsm[f"--{termdef.name}"] = (ws + atom_fsm[termdef.name]).reduce()

            print("reducing 2-length-FSMs")
            sequences: List[Tuple[str, str]] = []
            for key, fset in follow_set.items():
                if not isinstance(key, Terminal) or key.name not in atom_fsm:
                    continue
                for val in fset:
                    if val.name not in atom_fsm:
                        continue
                    sequences.append((key.name, val.name))
            print("1-length-FSMs:", len(self.fsm))
            print("2-length-FSMs:", len(sequences))

            print("confusion check")
            for term1, term2 in sequences:
                fsm1 = atom_fsm[term1]
                fsm2 = atom_fsm[term2]
                #if term1 in parser.ignore_tokens or term2 in parser.ignore_tokens or term1 == whitespace_key or term2 == whitespace_key:
                if term1 in parser.ignore_tokens or term2 in parser.ignore_tokens:
                    continue
                cont = fsm1 + fsm2
                for another_term, another in atom_fsm.items():
                    if not another.isdisjoint(cont):
                        if whitespace_key == "":
                            if self.strict_lex:
                                assert False, f"{term1} {term2} is confused with {another_term}."
                            else:
                                print(f"{term1} {term2} is confused with {another_term}.")
                        else:
                            # Attempt to put whitespace
                            self.confusion_termpairs.add((term1, term2))
                            cont_ws = fsm1 + self.fsm[f"--{term2}"]
                            if not another.isdisjoint(cont_ws):
                                if self.strict_lex:
                                    assert False, f"{term1} {term2} is confused with {another_term} even with whitespaces."
                                else:
                                    print(f"{term1} {term2} is confused with {another_term} even with whitespaces.")

            print("constructing 2-length-FSMs")
            for term1, term2 in sequences:
                if term1 in parser.ignore_tokens or term2 in parser.ignore_tokens:
                    continue
                if (term1, term2) in self.confusion_termpairs:
                    # must have spaces
                    self.fsm[f"{term1}-{term2}"] = (self.fsm[term1] + ws + atom_fsm[term2]).reduce()
                else:
                    # possibly spaces
                    self.fsm[f"{term1}-{term2}"] = (self.fsm[term1] + self.fsm[term2]).reduce()

            for name, fsm in tqdm(self.fsm.items(), desc="maskstore", total=len(self.fsm)):
                mask_store = get_dfa_maskstore(fsm, self.sorted_vocab)
                adjmat = get_dfa_adjmat(mask_store, self.sorted_vocab)
                distmat, _ = get_dfa_distmat(adjmat)
                INF = len(adjmat) + 1
                distmat = np.where(distmat == -1, INF, distmat)
                self.mask_store[name] = csr_array(mask_store + 1)
                self.acceptance_length[name] = distmat[:, list(fsm.finals)].min(axis=1)
                assert not any(self.acceptance_length[name] == INF)
                if DEBUG:
                    acc = get_minimum_acceptance_tokens(fsm, self.sorted_vocab, mask_store)
                    self.acceptance_tokens[name] = acc
            
            print("everything is ok.")
            with cachefile.open("wb") as f:
                pickle.dump({
                    "self": {key: getattr(self, key) for key in cachekeys},
                }, f)

        self.lark_lexer = parser._build_lexer(dont_ignore=True)
        self.ignore_types = frozenset(parser.lexer_conf.ignore)
 
    def clean_tokens(self, lextoks: List[lark.lexer.Token]) -> List[Terminal]:
        return [Terminal(t.type) for t in lextoks if t.type not in self.ignore_types]
    
    def partial_lex(self, text: str) -> Tuple[List[lark.lexer.Token], str]:
        """Lex input string as long as possible and return fixed lexical tokens and remainder."""
        lex_state = lark.lexer.LexerState(text)
        line_ctr = lex_state.line_ctr

        lexical_tokens = []
        reminder = ""
        while line_ctr.char_pos < len(lex_state.text):
            res = self.lark_lexer.match(lex_state.text, line_ctr.char_pos)
            if not res:
                reminder = lex_state.text[line_ctr.char_pos:]
                break
            value, type_ = res
            ignored = type_ in self.lark_lexer.ignore_types
            t = None
            if not ignored or type_ in self.lark_lexer.callback:
                t = lark.lexer.Token(type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
            line_ctr.feed(value, type_ in self.lark_lexer.newline_types)
            if t is not None:
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.lark_lexer.callback:
                    t = self.lark_lexer.callback[t.type](t)
                if not ignored:
                    if not isinstance(t, lark.lexer.Token):
                        raise "Callbacks must return a token (returned %r)" % t
                    lex_state.last_token = t
                    lexical_tokens.append(t)

        if len(lexical_tokens) > 0 and len(reminder) == 0:
            # last lexical token is unstable
            reminder = lexical_tokens[-1].value
            lexical_tokens = lexical_tokens[:-1]
        return lexical_tokens, reminder
    
    def yielding_lex(self, initial_text: str) -> Generator[Tuple[List[lark.lexer.Token], str], str, None]:
        """Lex input string as long as possible and return lexical tokens and remainder.
        You can append input by .send()
        """
        lex_state = lark.lexer.LexerState(initial_text)
        line_ctr = lex_state.line_ctr

        while True:
            lexical_tokens = []
            reminder = ""
            while line_ctr.char_pos < len(lex_state.text):
                res = self.lark_lexer.match(lex_state.text, line_ctr.char_pos)
                if not res:
                    reminder = lex_state.text[line_ctr.char_pos:]
                    break
                value, type_ = res
                ignored = type_ in self.lark_lexer.ignore_types
                t = None
                if not ignored or type_ in self.lark_lexer.callback:
                    t = lark.lexer.Token(type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
                line_ctr.feed(value, type_ in self.lark_lexer.newline_types)
                if t is not None:
                    t.end_line = line_ctr.line
                    t.end_column = line_ctr.column
                    t.end_pos = line_ctr.char_pos
                    if t.type in self.lark_lexer.callback:
                        t = self.lark_lexer.callback[t.type](t)
                    if not ignored:
                        if not isinstance(t, lark.lexer.Token):
                            raise "Callbacks must return a token (returned %r)" % t
                        lex_state.last_token = t
                        lexical_tokens.append(t)
            if len(lexical_tokens) > 0 and len(reminder) == 0:
                # last lexical token is unstable. rollback.
                reminder = lexical_tokens[-1].value
                lexical_tokens = lexical_tokens[:-1]
                line_ctr.char_pos -= len(reminder)
            addtional_text = yield lexical_tokens, reminder
            lex_state = lark.lexer.LexerState(lex_state.text + addtional_text, lex_state.line_ctr, lex_state.last_token)


def consume(dfa: interegular.FSM, start_state: int, input_str: str) -> Optional[int]:
    """Walk on dfa from start_state with input_str.
    If state goes a dead state, return None.
    If alive, return the state index.
    """
    state: Optional[int] = start_state
    for c in input_str:
        state = dfa.map[state].get(dfa.alphabet[c])
        if state is None:
            return None
    ## NOTE: Defined state is always alive.
    # assert dfa.islive(state)
    return state

def get_dfa_mask(dfa: interegular.FSM, start_state: int, vocab: List[Optional[str]]) -> np.ndarray:
    """Return vocab mask.
    If mask[i] == t >= 0, vocab[i] leads to a live state t. Otherwise -1
    """
    # everything is alive state
    assert all([dfa.islive(state) for state in dfa.states])
    mask = np.full(len(vocab), -1, dtype=np.int32)
    
    for tid, tok in enumerate(vocab):
        # sanity check of token
        if tok is None:
            continue
        next_state = consume(dfa, start_state, tok)
        if next_state is None:
            mask[tid] = -1
        else:
            mask[tid] = next_state
    return mask

def get_dfa_maskstore(dfa: interegular.FSM, vocab: List[Optional[str]]) -> np.ndarray:
    """Return vocab mask store.
    If mask[s,c] == t >= 0, vocab[c] leads to a live state t from state s. Otherwise -1.
    """
    assert len(dfa.states) == max(dfa.states) + 1  # states are 0..max
    maskstore = np.zeros((len(dfa.states), len(vocab)), dtype=np.int32)
    for start_state in dfa.states:
        maskstore[start_state, :] = get_dfa_mask(dfa, start_state, vocab)
    return maskstore

def get_dfa_adjmat(mask_store: np.ndarray, vocab: List[Optional[str]]) -> np.ndarray:
    """Return adjacency matrix.
    mat[i, j] == c means vocab[c] moves state i -> j. -1 means unreachable."""
    n_states = mask_store.shape[0]
    adjmat = np.full((n_states, n_states), -1, dtype=np.int32)
    for state in range(n_states):
        next_states = mask_store[state]  # live state j or otherwise -1
        for c, ns in enumerate(next_states):
            if ns >= 0:
                adjmat[state, ns] = c
    return adjmat

def get_dfa_distmat(adjmat, return_example=False) -> Tuple[np.ndarray, Optional[List[List[Optional[List[int]]]]]]:
    """Return distance matrix and shortest tokens.
    distances[i, j] == t >= 0 means you can feed t-length tokens to change state from i to j.
    And its example is shortest_paths[i][j]."""
    # Floyd-Warshall Algorithm
    num_states = len(adjmat)
    if return_example:
        shortest_paths = [[None for i in range(num_states)] for j in range(num_states)]
    INF = num_states + 1
    distances = np.full(adjmat.shape, INF, dtype=np.int32)  # initialize "Infinity" matrix

    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                distances[i, i] = 0
                if return_example:
                    shortest_paths[i][i] = []
            elif adjmat[i, j] >= 0:  # movable from i to j
                distances[i, j] = 1
                if return_example:
                    shortest_paths[i][j] = [adjmat[i, j].item()]

    for k in range(num_states):
        for i in range(num_states):
            for j in range(num_states):
                # test i -> k -> j
                if distances[i, k] + distances[k, j] < distances[i, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
                    if return_example:
                        shortest_paths[i][j] = shortest_paths[i][k] + shortest_paths[k][j]
    distances[distances == INF] = -1
    return distances, (shortest_paths if return_example else None)

def get_minimum_acceptance_tokens(dfa: interegular.FSM, vocab: List[Optional[str]], mask_store: np.ndarray) -> List[List[int]]:
    """shortest tokens from each state"""
    adjmat = get_dfa_adjmat(mask_store, vocab)
    distmat, paths = get_dfa_distmat(adjmat, return_example=True)
    result = []
    for start_state in range(len(distmat)):
        best = None
        for fini_state in dfa.finals:
            if distmat[start_state, fini_state] != -1:  # reachable
                candidate = paths[start_state][fini_state]
                if best is None or len(best) > len(candidate):
                    best = candidate
        assert best is not None  # check if our LLM can recover from any state
        result.append(best)
    return result
