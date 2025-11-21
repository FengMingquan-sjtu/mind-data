# mind-data

Utilities for recreating the math-informed synthetic dialogue pipeline described in the
paper.

## Data generation pipeline

- Loads the local `books_math_en_0527` parquet shards (decoded with the Qwen2 tokenizer).
- Chunks each document into overlapping 500-token windows (default overlap 50).
- Applies the seven conversational prompts from the paper to every chunk.
- Calls the DirectLLM-compatible API defined in `tmp.py` to obtain conversations.
- Filters out conversations shorter than 50 tokens before writing JSONL records.

## Setup

```bash
cd /cpfs/user/fengmingquan/mind-data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the API credentials (matching `tmp.py`) via environment variables or a local
`.env` file:

```
MIND_API_KEY=xxxxxxxx
MIND_BASE_URL=https://maas.devops.xiaohongshu.com/v1
MIND_LLM_MODEL=deepseek-r1
```

## Usage

Dry-run (no LLM calls, useful for plumbing tests):

```bash
python -m mind_data_pipeline.cli \
  --output-path data/mind_dry_run.jsonl \
  --max-samples 5 \
  --prompt-style two_students \
  --dry-run
```

Full generation with all prompt styles:

```bash
/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python -m mind_data_pipeline.cli \
  --output-path data/mind_conversations_3.jsonl \
  --max-samples 10 \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --model deepseek-v3 \
  --dataset-dir /prodcpfs/user/fengmingquan/dataset/raw/books_math_cn_0527
```

Refer to `python -m mind_data_pipeline.cli --help` for the complete list of
options (dataset split, streaming mode, temperature, etc.).