set -uex

python3 evaluation.py gemma --method original syncode xgrammar proposed
python3 evaluation.py llama --method original syncode xgrammar proposed

python3 evaluation.py gemma --limit 1.0 --hard_only --method syncode proposed
python3 evaluation.py gemma --limit 1.2 --hard_only --method syncode proposed
python3 evaluation.py gemma --limit 1.3 --hard_only --method syncode proposed
python3 evaluation.py gemma --limit 1.4 --hard_only --method syncode proposed
python3 evaluation.py gemma --limit 1.5 --hard_only --method syncode proposed

python3 evaluation.py gemma --normal_only --method outlines
python3 evaluation.py llama --normal_only --method outlines

### much time ###
#python3 evaluation.py gemma --hard_only --method outlines
#python3 evaluation.py llama --hard_only --method outlines
#################

python3 show_figure.py --folder outputs_gemma-2-2b-it_1.10
python3 show_table.py --folder outputs_gemma-2-2b-it_1.10
python3 show_table.py --folder outputs_Llama-2-7b-chat-hf_1.10
python3 show_limitations.py
