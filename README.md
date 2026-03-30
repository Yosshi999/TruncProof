# TruncProof

## Sample
```
$ pip install .
$ python sample_json_generation.py

#### sample input ####
<bos><start_of_turn>user
You are a helpful assistant that answers in JSON. Here’s the json schema you must adhere to:
<schema>
{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid", "securityProtocol", "bandwidth"]}
</schema>
I’m currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point’s SSID should be ’OfficeNetSecure’, it uses WPA2-Enterprise as its security protocol, and it’s capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details.<end_of_turn>


#### No Constraint ####
Proctime: 1.348 sec
New tokens: 30
*Note: I am not able to access your network or any specific hardware.*

```json
{"ssid": "OfficeNetSecure", "security

#### w/ Constraint ####
Proctime: 1.032 sec
New tokens: 30
{
"ssid": "OfficeNetSecure",
"securityProtocol": "WPA2-Enterprise",
"bandwidth": "130"}

#### MCTS w/ Constraint ####
Proctime: 14.579 sec
New tokens: 30


{"ssid": "OfficeNetSecure", "securityProtocol": "WPA2-Enterprise", "bandwidth": "1300 Mbps"}

```

# Reproducibility of IJCNN2026

## Setup
```
$ docker build -t truncproof .
$ docker run -it --rm --gpus all -v $PWD/ijcnn26:/workspace truncproof bash
```

Inside the container:

* Access to the models `meta-llama/Llama-2-7b-chat-hf` and `google/gemma-2-2b-it` is restricted. Run `huggingface-cli login` first.

## Experiments
All predictions are generated in the folder `outputs_*`

```
bash all_experiments.sh
```

### +prompt
```
python3 evaluation_compact.py gemma
python3 evaluation_compact.py llama

python3 show_table.py --folder outputs_gemma-2-2b-it_1.10_compact
python3 show_table.py --folder outputs_Llama-2-7b-chat-hf_1.10_compact
```


# Citation
```
@inproceedings{kato2026truncproof,
  title={{TruncProof}: A Guardrail for LLM-based JSON Generation under Token-Length Constraints},
  author={Kato, Yoshio and Shuhei, Tarashima},
  journal={The International Joint Conference on Neural Networks},
  year={2026}
}
```
