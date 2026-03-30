from __future__ import annotations
import lark
from lark import Lark
from lark.lark import LarkOptions
from lark.grammar import Terminal, NonTerminal, Symbol
from tqdm import tqdm

from dataclasses import dataclass
import sys
from typing import Optional, Tuple, List, Dict, Set, TypeVar, Generic
if sys.version_info.minor >= 9:
    from collections.abc import Sequence
else:
    from typing import Sequence
from itertools import combinations
from pathlib import Path
from logging import getLogger
import os
logger = getLogger(__name__)

VERBOSE = os.environ.get("VERBOSE", "0") == "1"

Expansion = Tuple[Symbol,...]
RuleMap = Dict[NonTerminal, List[Expansion]]

class Eps(Symbol):
    name = "__EPSILON__"
    is_term = True
    def __init__(self):
        pass

class Eos(Symbol):
    name = "__EOS__"
    is_term = True
    def __init__(self):
        pass

class Parser:
    """LL(1) Parser."""
    def __init__(self, text_grammar: str):
        LarkOptions._defaults["_plugins"] = {}  # clean up environment modified by Outlines
        parser = Lark(text_grammar, start="start", parser="lalr", lexer="basic", debug=True)
        start_names = parser.options.start
        assert len(start_names) == 1
        self.start_symbol = NonTerminal(start_names[0])

        rules = normalize_rules(parser.rules)
        #print_rules(rules)
        rules = remove_left_recursion(rules)
        rules = left_factoring(rules)
        #print_rules(rules)
        self.rules = rules
        self.terminals: List[Terminal] = [Terminal(termdef.name) for termdef in parser.terminals]
        self.nonterminals: List[NonTerminal] = list(rules.keys())

        first_set = {}
        for nonterm in (tqdm(rules.keys(), desc="FIRST of nonterm") if VERBOSE else rules.keys()):
            first_set[nonterm] = search_first_set([nonterm], rules, set(), first_set)
        for termdef in (tqdm(parser.terminals, desc="FIRST of term") if VERBOSE else parser.terminals):
            symbol = Terminal(termdef.name)
            first_set[symbol] = {symbol}
        
        follow_set = {}
        for nonterm in (tqdm(rules.keys(), desc="FOLLOW of nonterm") if VERBOSE else rules.keys()):
            follow_set[nonterm] = search_follow_set(nonterm, rules, set(), parser.options.start, first_set, follow_set)
        for termdef in (tqdm(parser.terminals, desc="FOLLOW of term") if VERBOSE else parser.terminals):
            symbol = Terminal(termdef.name)
            follow_set[symbol] = search_follow_set(symbol, rules, set(), parser.options.start, first_set, follow_set)
        
        director_map, conflicts = construct_director_map(rules, first_set, follow_set)
        for origin_name, e1, e2, buftop in conflicts:
            logger.error("LL(1) conflict in rule: {}\n-> {}\n-> {}\nreading: {}\n".format(
                origin_name, [e.name for e in e1], [e.name for e in e2], buftop)
            )
        if len(conflicts) == 0:
            logger.info("No conflicts fonud. This grammar is LL(1).")
        else:
            print("found", len(conflicts), "conflicts")
            raise
        
        self.first_set = first_set
        self.follow_set = follow_set
        self.table = get_parsing_table(director_map)
        #for key, val in self.table.items():
        #    print(key)
        #    for key2, val2 in val.items():
        #        print("|", key2, val2)

    def initialize(self) -> List[Symbol]:
        """Get initial stack. [0] is head"""
        return [self.start_symbol]

    def run(self, stack: List[Symbol], buffer: List[Terminal]) -> List[Symbol]:
        """Consume input buffer and return the result stack."""
        stack = stack.copy()
        cursor = 0
        while cursor < len(buffer):
            term = buffer[cursor]
            if len(stack) == 0 and term == Eos():
                assert cursor == len(buffer) - 1, "There must not be any symbol after EOS"
                return []  # reached EOS
            head = stack.pop(0)
            if head.is_term:
                assert head == term, f"Parse error at input[{cursor}]"
                cursor += 1
            else:
                assert term in self.table[head], f"Parse error (no available expansion) at input[{cursor}]"
                stack = list(self.table[head][term]) + stack
        return stack  # all inputs are consumed

    def run_if_possible(self, stack: List[Symbol], buffer: List[Terminal]) -> List[Symbol]:
        try:
            result = self.run(stack, buffer)
            return result, True
        except:
            return None, False

    def valid_inputs(self, stack: List[Symbol]) -> Set[Terminal|Eos]:
        if len(stack) == 0:
            return {Eos()}
        if stack[0].is_term:
            return {stack[0]}
        else:
            return set(self.table[stack[0]].keys())

def normalize_rules(rules: List[lark.grammar.Rule]) -> RuleMap:
    mapper: RuleMap = {}
    for rule in rules:
        origin = rule.origin.name
        if type(origin) != str:
            origin = origin.value
        origin = NonTerminal(origin)
        if origin not in mapper:
            mapper[origin] = []
        mapper[origin].append(tuple(rule.expansion))
    return mapper

def print_rules(rules: RuleMap):
    for origin, expansions in rules.items():
        origin = origin.name
        if not type(origin) == str:
            origin = origin.value
        print(f"{origin} ->", end="")
        for i, expansion in enumerate(expansions):
            rhs = ' '.join(e.name for e in expansion)
            if len(rhs) == 0:
                rhs = "ε"
            if i == 0:
                print(f" {rhs}")
            else:
                print(f"\t |{rhs}")

def new_unique_key(rules: RuleMap, base: str) -> str:
    """append prime to make unique name"""
    keys = set([k.name for k in rules.keys()])
    candidate = base
    while candidate in keys:
        candidate = base + "'"
    return candidate

def remove_left_recursion(rules: RuleMap) -> RuleMap:
    """from rule:
        A -> Aa | b
	to rule:
        A -> bA'
        A' -> aA' | eps
    """
    rules = rules.copy()
    for origin, expansions in list(rules.items()):
        new_rules: RuleMap = {}
        # find left recursion
        alphas: List[Expansion] = []
        betas: List[Expansion] = []
        for ex in expansions:
            if len(ex) > 0 and ex[0] == origin:
                alphas.append(ex[1:])
            else:
                betas.append(ex)
        if len(alphas) == 0:
            # no left recursion
            continue
        else:
            new_nonterminal = NonTerminal(new_unique_key(rules, origin.name))
            new_rules[origin] = []
            new_rules[new_nonterminal] = []
            for beta in betas:
                new_rules[origin].append((*beta, new_nonterminal))
            for alpha in alphas:
                new_rules[new_nonterminal].append((*alpha, new_nonterminal))
            new_rules[new_nonterminal].append(())
            rules.update(new_rules)
    return rules

T = TypeVar("T")
class TrieNode(Generic[T]):
    def __init__(self):
        self.leaf = False
        self.children: Dict[T, TrieNode] = {}
    def insert(self, word: Sequence[T]):
        ptr = self
        for char in word:
            if char not in ptr.children:
                ptr.children[char] = TrieNode()
            ptr = ptr.children[char]
        ptr.leaf = True


def left_factoring(rules: RuleMap) -> RuleMap:
    """from rule:
        A -> aBC|aBdE|f
	to rule:
        A->aBA'|f
        A'->C|dE
    """
    new_rules: RuleMap = {}
    for origin, expansions in rules.items():
        # construct trie of expansions. shared nodes can be factorized
        trie: TrieNode[Symbol] = TrieNode()
        for ex in expansions:
            trie.insert(ex)
        def walk(ptr: TrieNode, nonterm: NonTerminal, symbols: Expansion) -> List[Expansion]:
            """DFS to factorize trie.
            ptr: current node
            nonterm: origin of the expansions
            symbols: path from the root to the current node

            Returns the expansions after factorizing
            """
            if (ptr.leaf and len(ptr.children) >= 1) or len(ptr.children) > 1:
                # the path [root ... ptr] is factorable
                new_nonterminal = NonTerminal(nonterm.name + "/" + "_".join([s.name for s in symbols]))
                new_expansions = []  # rules of new_nonterminal -> *
                for key, child in ptr.children.items():
                    new_expansions.extend(walk(child, new_nonterminal, (key,)))
                if ptr.leaf:
                    new_expansions.append(())  # new_nonterminal -> ε
                new_rules[new_nonterminal] = new_expansions
                return [(*symbols, new_nonterminal)]
            else:
                result: List[Expansion] = []
                for key, child in ptr.children.items():
                    result.extend(walk(child, nonterm, symbols + (key,)))
                if ptr.leaf:
                    result.append(symbols)
                return result
        result: List[Expansion] = []
        for key, child in trie.children.items():
            result.extend(walk(child, origin, (key,)))
        if trie.leaf:
            result.append(())  # append epsilon
        new_rules[origin] = result
    return new_rules

def search_first_set(symbols: Sequence[Symbol], rules: RuleMap, visited: Set[Symbol], cache) -> Set[Terminal|Eps]:
    """Collect symbols of FIRST(α) in LL(1); {a: ∃β. α →* aβ}"""
    if len(symbols) == 0:
        return {Eps()}
    if len(symbols) > 1:
        result = set()
        for symbol in symbols:
            first = search_first_set([symbol], rules, visited, cache)
            result |= first - {Eps()}
            if Eps() not in first:
                break
        else:
            # there is an expansion α -> ε
            result.add(Eps())
        return result
    symbol = symbols[0]
    if symbol.is_term:
        return {symbol}
    # cache check
    if symbol in cache:
        return cache[symbol]
    # expand nonterminal
    if symbol in visited:
        # recursion detected
        return set()
    visited = visited | {symbol}  # stack self
    result = set()
    for expansion in rules[symbol]:
        result |= search_first_set(expansion, rules, visited, cache)
    return result

def search_follow_set(symbol: Symbol, rules: RuleMap, visited: Set[Symbol], start_symbols: Set[str], first_set: Dict[Symbol, Set[Terminal|Eps]], cache) -> Set[Terminal|Eos]:
    """Collect symbols of FOLLOW of A in LL(1); {a: ∃β,δ. S →* βAaδ}
        = {FIRST(α): ∃β. S →* βAα}
    """
    if symbol in visited:
        # recursion detected
        return set()
    if symbol in cache:
        return cache[symbol]
    visited = visited | {symbol}  # stack self
    result = set()
    if symbol.name in start_symbols:
        result |= {Eos()}
    for origin, rule in rules.items():
        for expansion in rule:
            for i in range(len(expansion)):
                if expansion[i] == symbol:
                    trailing_first = search_first_set(expansion[i+1:], rules, set(), first_set)
                    result |= trailing_first - {Eps()}
                    if Eps() in trailing_first:
                        f = search_follow_set(origin, rules, visited, start_symbols, first_set, cache)
                        cache[origin] = f
                        result |= f
    return result

def construct_director_map(rules: RuleMap, first_set: Dict[Symbol, Set[Terminal|Eps]], follow_set: Dict[Symbol, Set[Terminal|Eos]]):
    director_map = {}
    conflicts = []
    #for origin, expansions in tqdm(rules.items(), desc="DIRECTOR"):
    for origin, expansions in rules.items():
        director_map[origin] = {}
        for expansion in expansions:
            if len(expansion) > 0:
                if Eps() not in first_set[expansion[0]]:
                    director_map[origin][expansion] = first_set[expansion[0]]
                else:
                    director_map[origin][expansion] = first_set[expansion[0]] | follow_set[origin] - {Eps()}
            else:
                director_map[origin][expansion] = follow_set[origin]
        # disjoint check
        for e1, e2 in combinations(expansions, 2):
            f1 = director_map[origin][e1]
            f2 = director_map[origin][e2]
            if len(f1 & f2) > 0:
                conflicts.append((origin.name, e1, e2, f1 & f2))
    return director_map, conflicts

def get_parsing_table(director) -> Dict[NonTerminal, Dict[Terminal|Eos, Expansion]]:
    """Get parsing table for LL(1).
    dict is `nonterm on stack head`: NonTerminal => `terminal on input cursor`: Terminal => `expansion`: Expansion
    """
    table = {}
    for head, expansion_terms in director.items():
        table[head] = {}
        for expansion, terms in expansion_terms.items():
            for term in terms:
                table[head][term] = expansion
    return table

