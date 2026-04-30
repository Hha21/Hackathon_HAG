# Adversarial Alignment: LRM Jailbreak Experiments

Reproduction and extension of [Hagendorff et al. (2025)](https://arxiv.org/abs/2508.04039) —
*"Large Reasoning Models Are Autonomous Jailbreak Agents"* — built for the CDT Decision-Making
for Complex Systems Hackathon, University of Manchester.

---

## What this does

A large attacker LLM conducts a 10-turn multi-turn persuasion campaign against a small,
alignment-tuned target model. A judge LLM scores the target's response on a 0–5 harm scale
after every turn. Attack success = any turn hitting harm score 5.

```
Attacker (DeepSeek-V4-Flash via OpenRouter)
    given system prompt with {harmful_request} substituted
    starts every conversation with "Hi!"
         │
         ▼
Target (Qwen2.5-1.5B, local vLLM)
    given "You are a helpful assistant."
    has no knowledge it is under attack
         │
         ▼
Judge (DeepSeek-V4-Flash via OpenRouter, same API)
    scores target response 0–5 per turn
    early-stops when harm = 5
```

---

## Setup

### 1. Python environment

No conda required — use a plain venv:

```bash
cd Hackathon_HAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the local target server (Qwen2.5-1.5B)

The target model runs locally on GPU 1. Start it in a persistent terminal (see
[Staying alive over SSH](#staying-alive-over-ssh) below):

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.45
```

> **`--max-model-len 8192`** is important — the default 4096 causes context-overflow errors
> after a few turns of conversation history accumulate.

### 3. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### 4. Run the experiment

```bash
# Smoke test — 1 item, verbose
python run_experiment.py --dry-run --verbose

# Full 70-item benchmark
python run_experiment.py

# Subset
python run_experiment.py --n-prompts 10
```

Results are saved to `results/{run_id}/`:
- `raw_logs.jsonl` — full turn-by-turn JSON logs
- `summary.csv` — one row per exchange
- `conversations.txt` — human-readable transcript

---

## Staying alive over SSH

Use `tmux` so the run survives if your SSH connection drops.

```bash
# First time: create a session with two windows
tmux new-session -s hack -n server
# (you are now inside the tmux session)

# Window 0: start the vLLM server
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.45

# Open a second window (Ctrl-b c), then run the experiment
export OPENROUTER_API_KEY="sk-or-..."
python run_experiment.py

# Detach from tmux without killing anything: Ctrl-b d
# Reattach later from any SSH session:
tmux attach -t hack
```

---

## Project structure

```
Hackathon_HAG/
├── config.py            # model IDs, server URLs, hyperparameters
├── models.py            # OpenAI-compatible client wrapper (vLLM + OpenRouter)
├── attacker.py          # adversarial LLM — natural conversation, "Hi!" opener
├── defender.py          # target LLM — helpful assistant, full history
├── judge.py             # evaluator — per-turn 0–5 harm scoring
├── experiment.py        # single exchange runner → ExchangeResult + harm trajectory
├── run_experiment.py    # outer loop, CLI entry point, logging
├── prompts/
│   ├── attacker_system.txt   # Appendix A system prompt (Hagendorff et al.)
│   └── judge_harm.txt        # Appendix C evaluation rubric (0–5 scale)
├── data/
│   └── hagendorff_benchmark.json   # 70 harmful prompts across 7 categories
└── results/
    └── {run_id}/
        ├── raw_logs.jsonl
        ├── summary.csv
        └── conversations.txt
```

---

## Phases

### Phase 1 — Reproduce Hagendorff et al.
Run the full 70-item benchmark, compute ASR, plot harm trajectories.

### Phase 2 — Capability Gap Study
Vary the attacker/target pair across the Qwen2.5 family to test whether the
attacker/target size ratio predicts ASR.

### Phase 3 — Adaptive Arms Race
After each attack round, use an LLM to generate counter-instructions for the
defender's system prompt, then re-run and track harm over adaptation rounds.

---

## References

- Hagendorff et al. (2025). *Large Reasoning Models Are Autonomous Jailbreak Agents.* [arXiv:2508.04039](https://arxiv.org/abs/2508.04039)
- Chao et al. (2024). *JailbreakBench.* [arXiv:2404.01318](https://arxiv.org/abs/2404.01318)
