from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time

from truncproof import GrammarLogitsProcessor
from truncproof.generation_utils import mct_search


repo = "google/gemma-2-2b-it"
tokenizer_option = {}
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

tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", **tokenizer_option)
model = model.to("cuda")

prompt = [{
    "content":
        """You are a helpful assistant that answers in JSON. Here’s the json schema you must adhere to:
<schema>
{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid", "securityProtocol", "bandwidth"]}
</schema>
I’m currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point’s SSID should be ’OfficeNetSecure’, it uses WPA2-Enterprise as its security protocol, and it’s capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details.""",
    "role": "user",
}]

print("eos (tok):", tokenizer.eos_token_id)
print("eos (model):", model.config.eos_token_id)

print("#### sample input ####")
print(tokenizer.apply_chat_template(prompt, tokenize=False))
print()

max_new_tokens = 30
with torch.no_grad():
    input_ids = torch.LongTensor(
        tokenizer.apply_chat_template(prompt, tokenize=True)
    ).to(model.device).unsqueeze(0)  # (batchsize, sequence)
    input_length = input_ids.size(1)

    print("#### No Constraint ####")
    tic = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    output = tokenizer.decode(output_ids.tolist()[0][input_length:], skip_special_tokens=True)
    tac = time.perf_counter()
    print(f"Proctime: {tac - tic:.3f} sec")
    print("New tokens:", len(output_ids[0]) - input_length)
    print(output)
    print()

    print("#### w/ Constraint ####")
    logiproc = GrammarLogitsProcessor(
        input_length + max_new_tokens,
        text_grammar, "", tokenizer, model.config.eos_token_id, input_length
    )

    tic = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=[logiproc],
    )
    output = tokenizer.decode(output_ids.tolist()[0][input_length:], skip_special_tokens=True)
    tac = time.perf_counter()
    print(f"Proctime: {tac - tic:.3f} sec")
    print("New tokens:", len(output_ids[0]) - input_length)
    print(output)
    print()

    print("#### MCTS w/ Constraint ####")
    logiproc = GrammarLogitsProcessor(
        input_length + max_new_tokens,
        text_grammar, "", tokenizer, model.config.eos_token_id, input_length
    )

    tic = time.perf_counter()
    output_ids = mct_search(
        model,
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer = tokenizer,
        logits_processor=logiproc,
        n_trials=20,
        puct_coeff=5,
        max_new_tokens=max_new_tokens
    )
    output = tokenizer.decode(output_ids.tolist()[0][input_length:], skip_special_tokens=True)
    tac = time.perf_counter()
    print(f"Proctime: {tac - tic:.3f} sec")
    print("New tokens:", len(output_ids[0]) - input_length)
    print(output)
    print()
