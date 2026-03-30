"""
Monte Carlo Tree Search for transformers==4.51.3
"""
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
from copy import deepcopy
import os

import torch
from transformers import (
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    logging,
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HQQQuantizedCache,
    HybridCache,
    MambaCache,
    OffloadedCache,
    QuantizedCacheConfig,
    QuantoQuantizedCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.utils.import_utils import is_torchdynamo_compiling
from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING

logger = logging.get_logger(__name__)

VERBOSE = os.environ.get("VERBOSE", "0") == "1"

class HookedCache(Cache):
    """KV-Cache with logging states. Used in MCTS, which frequently re-generates the sequence."""
    def __init__(self, cache: Cache, strict: bool = False):
        super().__init__()
        self.base_cache = cache
        self.previous_query = []
        self.strict = strict
    def reset_mem(self):
        self.previous_query = []
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0 and self.strict:
            assert len(self.previous_query) == 0, "Cache must be empty"
        self.previous_query.append({
            "key_states": key_states,
            "value_states": value_states,
            "layer_idx": layer_idx,
            "cache_kwargs": cache_kwargs,
        })
        return self.base_cache.update(key_states, value_states, layer_idx, cache_kwargs)
    def get_seq_length(self, layer_idx=0):
        return self.base_cache.get_seq_length(layer_idx)
    def get_max_length(self):
        return self.base_cache.get_max_length()
    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.base_cache.get_usable_length(new_seq_length, layer_idx)
    def reorder_cache(self, beam_idx):
        return self.base_cache.reorder_cache(beam_idx)
    @property
    def seen_tokens(self):
        return self.base_cache.seen_tokens

@dataclass
class MCTNode:
    child_nodes: Dict[int, "MCTNode"] = field(default_factory=dict)
    rollout_ppl: Optional[float] = None  # the cached value if tree is roll-outed from this node
    # these fields below are assigned once this node is expanded
    cache_queries: List[dict] = field(default_factory=list)
    #child_tokens: Optional[torch.LongTensor] = None  # (1, vocab)
    child_probas: Optional[torch.FloatTensor] = None  # (1, vocab)
    child_accum: Optional[torch.FloatTensor] = None  # accumulated negppl (1, vocab)
    child_visited: Optional[torch.LongTensor] = None  # (1, vocab)
    child_nll: Optional[torch.FloatTensor] = None  # negative log likelihood (1, vocab)
    def is_expanded(self):
        return self.child_visited is not None
    def get_puct(self, puct_coeff: float):  # -> (1, vocab)
        # return score (higher is better) based on negppl (higher is better)
        return (
            self.child_accum +
            puct_coeff * self.child_probas * self.child_visited.sum().sqrt() / (1 + self.child_visited)
        )


@torch.no_grad()
def mct_search(
    self: GenerationMixin, 
    inputs: torch.Tensor,
    n_trials: int,
    puct_coeff: float,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    streamer: Optional["BaseStreamer"] = None,
    tokenizer = None,
    **kwargs,
):
    """based on v4.51.3"""
    self._validate_model_class()
    assert not self.config.is_encoder_decoder
    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config, **kwargs
    )
    self._validate_model_kwargs(model_kwargs.copy())
    # assistant not supported

    logits_processor = logits_processor or LogitsProcessorList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
    if VERBOSE:
        print(accepts_attention_mask, requires_attention_mask, kwargs_has_attention_mask)


    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    assert model_input_name == "input_ids"
    assert batch_size == 1
    input_ids = inputs_tensor  # LongTensor (1, S)

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    eos_token_id = generation_config._eos_token_tensor.item()

    assert generation_config.use_cache
    model_kwargs["use_cache"] = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )

    if streamer is not None:
        streamer.put(input_ids.cpu())

    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    assert "mamba" not in self.__class__.__name__.lower()

    if generation_config.cache_implementation is None:
        assert self._supports_default_dynamic_cache()
        initialize_cache = lambda: DynamicCache()
    else:
        assert generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING
        assert generation_config.num_beams == 1, "Beam search is not supported"
        assert generation_config.num_return_sequences == 1, "Only single sequence is supported"
        initialize_cache = lambda: self._get_cache(
            cache_implementation=generation_config.cache_implementation,
            batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
            max_cache_len=generation_config.max_length,
            device=device,
            model_kwargs=model_kwargs,
        )
    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    max_length = generation_config.max_length
    assert not generation_config.output_attentions
    assert not generation_config.output_hidden_states

    tree = MCTNode()
    prebuilt_nodes = 0
    prompt_ids = input_ids.clone()
    prebuilt_nll = 0
    prebuilt_queries: List[dict] = []
    while input_ids.shape[1] < max_length:
        for trial_iter in range(prebuilt_nodes, n_trials):
            selection_tree = tree
            simulation_input_ids = input_ids.clone()  # (1, S)
            simulation_nll = prebuilt_nll
            simulation_cache = initialize_cache()
            for q in prebuilt_queries:
                simulation_cache.update(**q)
            for q in tree.cache_queries:
                simulation_cache.update(**q)
            selection_path = []
            int_token_id = None
            # Selection
            while selection_tree.is_expanded():
                puct = selection_tree.get_puct(puct_coeff)  # (1, vocab)
                #if len(selection_path) == 1:
                #    pk = puct[0].topk(5)
                #    print("selection PUCT", pk.values)
                #    print("token", tokenizer.convert_ids_to_tokens(pk.indices))
                #    print("visited", selection_tree.child_visited[0, pk.indices])
                #    print("accum", selection_tree.child_accum[0, pk.indices])
                #    print("probas", selection_tree.child_probas[0, pk.indices])
                #    raise
                token_id = torch.argmax(puct, dim=-1, keepdim=True)  # (1, 1)
                int_token_id = token_id.item()
                selection_path.append(int_token_id)
                simulation_nll += selection_tree.child_nll[:, int_token_id].item()
                simulation_input_ids = torch.cat([simulation_input_ids, token_id], dim=-1)
                if int_token_id not in selection_tree.child_nodes:
                    selection_tree.child_nodes[int_token_id] = MCTNode()
                for q in selection_tree.child_nodes[int_token_id].cache_queries:
                    simulation_cache.update(**q)
                selection_tree = selection_tree.child_nodes[int_token_id]
            if int_token_id == eos_token_id or simulation_input_ids.shape[1] >= max_length:
                # Unable to expand.
                ppl = math.exp(simulation_nll / (simulation_input_ids.shape[1] - prompt_ids.shape[1]))
            else:
                # Expansion and Simulation
                model_kwargs["past_key_values"] = simulation_cache
                if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
                    model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                        simulation_input_ids, generation_config, model_kwargs
                    )
                model_kwargs = self._get_initial_cache_position(simulation_input_ids, model_kwargs)
                model_inputs = self.prepare_inputs_for_generation(simulation_input_ids, **model_kwargs)
                model_inputs["past_key_values"] = HookedCache(model_inputs["past_key_values"], strict=True)
                outputs = self(**model_inputs, return_dict=True)

                selection_tree.cache_queries = outputs.past_key_values.previous_query

                next_token_logits = outputs.logits[:, -1, :].clone()
                mod_next_token_scores = logits_processor(simulation_input_ids, next_token_logits)
                mod_next_token_probs = torch.nn.functional.softmax(mod_next_token_scores / 2.0, dim=-1)
                selection_tree.child_probas = mod_next_token_probs
                selection_tree.child_accum = torch.zeros_like(mod_next_token_probs)
                selection_tree.child_visited = torch.zeros_like(mod_next_token_probs).long()
                selection_tree.child_nll = -next_token_logits.log_softmax(-1)

                # Start simulation (greedy rollout from `selection_tree`)
                token_id = torch.argmax(mod_next_token_scores, dim=-1, keepdim=True)
                int_token_id = token_id.item()
                selection_path.append(int_token_id)
                simulation_tree = MCTNode()
                selection_tree.child_nodes[int_token_id] = simulation_tree  # assign rollout_ppl afterward
                if selection_tree.rollout_ppl is not None:
                    ppl = selection_tree.rollout_ppl
                else:
                    simulation_input_ids = torch.cat([simulation_input_ids, token_id], dim=-1)
                    simulation_nll += selection_tree.child_nll[:, int_token_id].item()
                     
                    outputs.past_key_values = outputs.past_key_values.base_cache  # deactivate hook
                    while int_token_id != eos_token_id and simulation_input_ids.shape[1] < max_length:
                        model_kwargs = self._update_model_kwargs_for_generation(
                            outputs,
                            model_kwargs,
                            is_encoder_decoder=self.config.is_encoder_decoder,
                        )
                        model_inputs = self.prepare_inputs_for_generation(simulation_input_ids, **model_kwargs)
                        outputs = self(**model_inputs, return_dict=True)
                        next_token_logits = outputs.logits[:, -1, :].clone()
                        mod_next_token_scores = logits_processor(simulation_input_ids, next_token_logits)
                        token_id = torch.argmax(mod_next_token_scores, dim=-1, keepdim=True)
                        #token_id = torch.multinomial(mod_next_token_scores.softmax(-1), num_samples=1)
                        int_token_id = token_id.item()
                        simulation_input_ids = torch.cat([simulation_input_ids, token_id], dim=-1)
                        simulation_nll += -next_token_logits.log_softmax(-1)[:, int_token_id].item()
                    ppl = math.exp(simulation_nll / (simulation_input_ids.shape[1] - prompt_ids.shape[1]))
                simulation_tree.rollout_ppl = ppl

            # print(f"#### [{trial_iter}] ppl: {ppl:.3f}",
            #     tokenizer.convert_ids_to_tokens(input_ids[0, prompt_ids.shape[1]:].cpu().numpy()),
            #     tokenizer.convert_ids_to_tokens(selection_path),
            #     "".join(tokenizer.convert_ids_to_tokens(simulation_input_ids[0, prompt_ids.shape[1]:].cpu().numpy())))

            # Backpropagation
            propagation_tree = tree
            for idx in selection_path[:-1]:
                propagation_tree.child_visited[:, idx] += 1
                #propagation_tree.child_accum[:, idx] += 1 / ppl
                propagation_tree.child_accum[:, idx] = max(propagation_tree.child_accum[:, idx].item(), 1 / ppl)
                propagation_tree = propagation_tree.child_nodes[idx]
            propagation_tree.child_visited[:, selection_path[-1]] += 1
            propagation_tree.child_accum[:, selection_path[-1]] = max(propagation_tree.child_accum[:, selection_path[-1]].item(), 1 / ppl)
        # Decide token
        next_tokens = tree.child_accum.argmax(-1)  # (1, 1)
        int_next_token = next_tokens.item()
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        if int_next_token == eos_token_id:
            break
        next_tree = tree.child_nodes[int_next_token]
        prebuilt_nodes = tree.child_visited[:, int_next_token].item()
        prebuilt_nll += tree.child_nll[:, int_next_token].item()
        prebuilt_queries.extend(tree.cache_queries)
        tree = next_tree
    return input_ids

