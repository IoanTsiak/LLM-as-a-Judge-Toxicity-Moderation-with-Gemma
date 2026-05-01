# CSE 528 - LLM-as-a-Judge Toxicity Moderation

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements_final.txt
hf auth login
```

## Demo

```powershell
python tox_judge_fixed_final.py --model google/gemma-2-2b-it demo --method A --text "you are stupid"
python tox_judge_fixed_final.py --model google/gemma-2-2b-it demo --method B --text "you are stupid"
python tox_judge_fixed_final.py --model google/gemma-2-2b-it demo --method C --self_n 5 --text "you are stupid"
```

## Dataset format

CSV with columns:

```csv
text,label
"you are stupid",1
"have a nice day",0
```

Accepted text columns: `text`, `comment`, `comment_text`, `content`  
Accepted label columns: `label`, `toxic`, `target`

## Small test runs first

```powershell
python tox_judge_fixed_final.py --model google/gemma-2-2b-it eval --dataset data\jigsaw_subset.csv --method A --sample_n 20 --out_dir outputs_A_test
python tox_judge_fixed_final.py --model google/gemma-2-2b-it eval --dataset data\jigsaw_subset.csv --method B --sample_n 20 --out_dir outputs_B_test
python tox_judge_fixed_final.py --model google/gemma-2-2b-it eval --dataset data\jigsaw_subset.csv --method C --self_n 3 --sample_n 20 --out_dir outputs_C_test
```

## Final runs

```powershell
python tox_judge_fixed_final.py --model google/gemma-2-2b-it eval --dataset data\jigsaw_subset.csv --method A --sample_n 50 --out_dir outputs_A
python tox_judge_fixed_final.py --model google/gemma-2-2b-it eval --dataset data\jigsaw_subset.csv --method B --sample_n 50 --out_dir outputs_B
python tox_judge_fixed_final.py --model google/gemma-2-2b-it eval --dataset data\jigsaw_subset.csv --method C --self_n 3 --sample_n 50 --out_dir outputs_C
```

## Robustness

```powershell
python tox_judge_fixed_final.py --model google/gemma-2-2b-it robustness --dataset data\jigsaw_subset.csv --method A --sample_n 20 --out_dir outputs_robust_A
```

## Consistency

```powershell
python tox_judge_fixed_final.py --model google/gemma-2-2b-it consistency --dataset data\jigsaw_subset.csv --method C --self_n 3 --sample_n 20 --seeds 11 22 33 --out_dir outputs_consistency_C
```
