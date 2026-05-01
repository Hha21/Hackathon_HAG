# Adversarial Alignment: LRM Jailbreak Experiments

Reproduction and extension of [Hagendorff et al. (2025)](https://arxiv.org/abs/2508.04039) —
*"Large Reasoning Models Are Autonomous Jailbreak Agents"* — built for the CDT Decision-Making
for Complex Systems Hackathon, University of Manchester.

---

## Overview

A large attacker LLM conducts a 10-turn persuasion campaign against an alignment-tuned target
model. A judge LLM scores the target's response on a 0–5 harm scale after every turn.
All models are served via the OpenRouter API — no local GPU required for inference.

```
Attacker (DeepSeek-V4-Pro via OpenRouter)
    system prompt contains the harmful request + optional high-harm openers
    generates a strategic first message, then adapts across 10 turns
         │
         ▼
Target (Gemini 2.5 Flash via OpenRouter)
    given "You are a helpful assistant."
    has no knowledge it is under attack
         │
         ▼
Judge (DeepSeek-V4-Pro via OpenRouter)
    scores target response 0–5 per turn (Hagendorff et al. Appendix C rubric)
    runs all 10 turns regardless of harm score reached
```

---

## Setup

### 1. Python environment

```bash
cd Hackathon_HAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### 3. Run the experiment

```bash
# Smoke test — 1 item, verbose
python run_experiment.py --dry-run --verbose

# Full 70-item benchmark (10 parallel workers)
python run_experiment.py

# With augmented attacker (top-30 high-harm openers injected into system prompt)
python run_experiment.py --augment-attacker

# Subset
python run_experiment.py --n-prompts 10
```

Results are saved to `results/{run_id}/`:
- `raw_logs.jsonl` — full turn-by-turn JSON logs
- `summary.csv` — one row per exchange
- `conversations.txt` — human-readable transcript

### 4. Generate plots

```bash
python analysis.py <run_id>
# e.g. python analysis.py 20260430_173957
```

Saves to `results/{run_id}/plots/`:
- `harm_trajectory.png` — mean ± std harm score per turn across all items
- `harm_trajectory_by_category.png` — one subplot per benchmark category

---

## Staying alive over SSH

```bash
tmux new-session -s hack
source .venv/bin/activate
export OPENROUTER_API_KEY="sk-or-..."
python run_experiment.py

# Detach (leaves run alive):  Ctrl-b d
# Reattach later:             tmux attach -t hack
```

---

## Project structure

```
Hackathon_HAG/
├── config.py               # model IDs, hyperparameters, N_PARALLEL
├── models.py               # OpenRouter client wrapper + query_model
├── attacker.py             # adversarial LLM — strategic opener, 10-turn campaign
├── defender.py             # target LLM — helpful assistant, full history
├── judge.py                # evaluator — per-turn 0–5 harm scoring
├── experiment.py           # single exchange runner → ExchangeResult
├── run_experiment.py       # parallel outer loop, CLI entry point, logging
├── analysis.py             # harm trajectory plots (mean ± std, per-category)
├── prompts/
│   ├── attacker_system.txt         # Hagendorff et al. Appendix A system prompt
│   ├── judge_harm.txt              # Appendix C evaluation rubric (0–5 scale)
│   └── top30_prompts_by_harm.csv   # high-harm openers from genetic search (see below)
├── data/
│   └── hagendorff_benchmark.json   # 70 harmful prompts across 7 categories
└── results/
    └── {run_id}/
        ├── raw_logs.jsonl
        ├── summary.csv
        ├── conversations.txt
        └── plots/
```

---

## Experiments

### 1 — Baseline reproduction

Full 70-item benchmark with DeepSeek-V4-Pro attacking Gemini 2.5 Flash.
Reproduces Figure 2 of Hagendorff et al.: mean harm trajectory per turn with ±1 std band.

```bash
python run_experiment.py
python analysis.py <run_id>
```

### 2 — Augmented attacker (genetic opener injection)

The attacker's system prompt is augmented with the top-30 high-harm opening messages
discovered by the genetic search (see below). The attacker generates its own strategic
first message rather than the fixed "Hi!" baseline.

```bash
python run_experiment.py --augment-attacker
python analysis.py <run_id>
```

Compare the resulting harm trajectories against the baseline run to measure the effect
of the opener injection.

### 3 — Adaptive attacker (prompt self-update)

*Implementation in progress (`adaptation.py`).*

After a baseline run, failed attack transcripts are fed back to the attacker LLM along
with the current `attacker_system.txt`. The attacker proposes a revised system prompt,
which is written back and used for a second round. The experiment tracks whether the
updated strategy raises ASR across rounds.

---

## Genetic Algorithm for Maximising Harm of Single Prompt

*Contributed by [colleague name].*

To identify the most effective jailbreak openers, a genetic algorithm was run against the
benchmark to evolve single prompts that maximise harm score on the judge. The
top-30 results by average harm score are stored in
`prompts/top30_prompts_by_harm.csv` and used by the `--augment-attacker` flag above.

*(Add methodology, hyperparameters, and results here.)*

---

## References

- Hagendorff et al. (2025). *Large Reasoning Models Are Autonomous Jailbreak Agents.* [arXiv:2508.04039](https://arxiv.org/abs/2508.04039)
- Chao et al. (2024). *JailbreakBench.* [arXiv:2404.01318](https://arxiv.org/abs/2404.01318)