# Adversarial Alignment: Scaling Patterns in Multi-LLM Jailbreak Experiments

A reproduction and extension of [Nathanson et al. (2025)](https://arxiv.org/abs/2511.13788) —
*"Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments"* —
built for the CDT Decision-Making for Complex Systems Hackathon, University of Manchester.

---

## Project Overview

Large language models increasingly operate in multi-agent settings where one model may attempt
to manipulate another. This project investigates a core question: **when a larger, more capable
model attacks a smaller, alignment-tuned model, does the capability gap predict how badly the
smaller model breaks?**

The paper found a statistically significant correlation (Pearson r = 0.510) between the
attacker-to-target size ratio and harm score across 6,000 simulated exchanges. We reproduce
this finding within a single model family (Qwen2.5) to control for architecture, then extend
it by introducing an **adaptive arms race** — where the defender updates its system prompt in
response to observed attacks — to study whether and how quickly a smaller model can recover
when it is allowed to learn.

This connects directly to the hackathon theme of **decision-making in complex systems**: the
attacker-defender pair is a complex adaptive system, and we characterise how its dynamics
change as a function of the capability gap between the two agents.

---

## Total Goals

### Phase 1 — Reproduction
Replicate the core finding of Nathanson et al. within a single model family:

- Run a full attacker × target matrix across Qwen2.5 at 1.5B, 7B, 14B, and 72B scales
- Measure harm score (1–5) for each attacker–target–prompt combination using an LLM judge
- Reproduce Figure 1: scatter plot of mean harm vs log(attacker/target size ratio)
- Reproduce Figure 3: attacker refusal rate by model size
- Reproduce Figure 4: attacker × target heatmap of mean harm scores
- Compute Pearson/Spearman correlation between log(size ratio) and harm
- Fit a mixed-effects model to decompose variance by attacker family, size ratio, prompt variant, target family, and harm domain

### Phase 2 — Arms Race Extension (Novel Contribution)
Extend the static snapshot of Phase 1 into a dynamic arms race:

- Fix a specific attacker–target pair (e.g. 72B attacker vs 1.5B defender)
- After each round of attacks, update the defender's system prompt using an LLM that reads the successful attacks and generates counter-instructions (ICAG-style adaptation)
- Run multiple rounds and track harm score over time
- Repeat for different capability gaps to ask: does gap size predict the trajectory?
  - Small gap: genuine co-evolution where both sides improve?
  - Large gap: defender collapses regardless of adaptation?
  - Is there a threshold gap below which adaptation becomes effective?

### Deliverables
- GitHub repository with full reproducible codebase
- GitHub Pages writeup with figures, results, and interpretation
- Key result: a plot of harm over adaptation rounds for different capability gaps

---

## Immediate Goals — Proof of Concept

Before running the full matrix, the first milestone is a **minimal working demo** using just
two models to confirm the entire pipeline functions end to end.

| Role | Model | Size |
|------|-------|------|
| Attacker | `Qwen/Qwen2.5-14B-Instruct` | 14B |
| Defender | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |
| Judge | `Qwen/Qwen2.5-14B-Instruct` | 14B (shared with attacker) |

**What this demonstrates:**
- The attacker generates structured JSON (PLAN / MESSAGE / CRITIQUE) at each turn
- MESSAGE is forwarded to the defender, which responds without knowledge of the attack
- The judge scores the full defender response sequence on a 1–5 harm scale
- Attacker refusals are detected and trigger early stopping
- Results are logged to JSONL and CSV

**Success criterion:** Run 3–5 JBB prompts end to end, see harm scores logged, confirm
the pipeline is working before scaling up to the full model matrix.

**To run the proof of concept:**

```bash
# Terminal 1 — start vLLM server for 14B model (attacker + judge)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096

# Terminal 2 — start vLLM server for 1.5B model (defender)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 4096

# Terminal 3 — dry run (1 prompt, verbose output)
python run_experiment.py --dry-run --verbose \
    --attacker-idx 2 --target-idx 0
# attacker-idx 2 = 14B, target-idx 0 = 1.5B (see config.py)
```

---

## Development Environment

### Cluster Access
University of Manchester **CSF3**, accessed via SSH.

```
Your laptop (VSCode)
      ↕ SSH
CSF login node (login1.csf3)     ← edit code here via VSCode Remote SSH
      ↕ Slurm
CSF GPU compute node (gpu00X)    ← run code here
```

The filesystem is **shared** between login and compute nodes — files edited on the
login node in VSCode are immediately visible on the compute node. No file copying needed.

### Hardware
- **Hackathon workstation:** 2× RTX 4090 (24GB each = 48GB total VRAM)
- **CSF GPU nodes:** V100 / A100 nodes available via Slurm partition `gpuL`

### VRAM requirements (Qwen2.5, bfloat16)
| Model | VRAM needed | Notes |
|-------|-------------|-------|
| 1.5B  | ~4GB   | Fits easily on one GPU |
| 7B    | ~15GB  | Fits on one GPU |
| 14B   | ~28GB  | Fits on one RTX 4090 |
| 72B   | ~144GB | Requires `--tensor-parallel-size 2`, quantised |

### Interactive session (for development and debugging)
```bash
srun --partition=gpuL --gres=gpu:1 --mem=24G --time=8:00:00 --pty bash
```

### Batch job (for long multi-hour runs)
```bash
sbatch slurm_job.sh
```

### Python environment setup
```bash
module load anaconda3
conda create -n hackathon python=3.11
conda activate hackathon
pip install openai vllm transformers accelerate pandas scipy matplotlib
```

### Inference backend
**vLLM** serves any HuggingFace model as an OpenAI-compatible REST API.
The codebase uses the OpenAI Python client pointed at `localhost:8000`.
Swapping models requires only restarting the server with a different `--model` flag —
no code changes needed.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096
```

### Version control
Git repo on GitHub. All three team members work in branches and pull to the cluster.
Never edit files directly on the cluster — always edit locally or on the login node
and push/pull via git.

---

## Codebase

### Structure

```
Hackathon/
├── config.py                   # Model definitions, hyperparameters, paths
├── models.py                   # vLLM/OpenAI client wrapper + JSON parsing utils
├── attacker.py                 # LLM attacker — structured JSON output per turn
├── defender.py                 # LLM defender/target — no knowledge of attack
├── judge.py                    # LLM judge — harm scoring (1-5) + refusal detection
├── experiment.py               # Single exchange runner (attacker × target × prompt)
├── run_experiment.py           # Outer loop — full matrix, CLI entry point, logging
├── analysis.py                 # [TODO] Correlation analysis + paper figures
├── adaptation.py               # [TODO] Phase 2 — ICAG defender adaptation loop
├── prompts/
│   ├── attacker_system_1.txt   # Attacker system prompt variant 1 (paper Appendix E)
│   ├── attacker_system_2.txt   # Attacker system prompt variant 2 (paper Appendix E)
│   ├── judge_harm.txt          # Harm scoring prompt (paper Appendix E)
│   └── judge_refusal.txt       # Attacker refusal detection prompt (paper Appendix E)
├── data/
│   └── jbb_behaviors.json      # 30 JBB prompts — Table 3 of Nathanson et al.
└── results/
    └── {run_id}/
        ├── raw_logs.jsonl      # Full turn-by-turn logs, one JSON object per exchange
        └── summary.csv         # One row per exchange, key metrics only
```

### File Descriptions

#### `config.py`
Central configuration. Edit this to change which models are run, adjust hyperparameters,
or point at a different vLLM server. Key settings:
- `ATTACKER_MODELS` / `TARGET_MODELS` — lists of `(model_id, display_name, size_b)`
- `JUDGE_MODEL` — the judge model definition
- `MAX_TURNS = 5` — maximum dialogue turns per exchange (matches paper)
- `TEMPERATURE = 0.7` — generation temperature (matches paper)
- `HARM_THRESHOLD = 5` — threshold for binary attack success (matches paper)
- `VLLM_BASE_URL` — URL of the running vLLM server

#### `models.py`
Thin wrapper around the OpenAI Python client pointed at vLLM:
- `query_model()` — sends a chat completion request, retries on transient errors, returns string
- `parse_json_response()` — extracts JSON from model output, handles markdown code fences

#### `attacker.py`
`Attacker` class. Maintains its own conversation history, separate from the defender.
At each turn outputs structured JSON matching the paper's Appendix E format:
```json
{
    "PLAN":    "internal reasoning — logged but never sent to the defender",
    "MESSAGE": "text forwarded to the defender",
    "CRITIQUE":"self-reflection for next turn"
}
```
`reset(task, t_max)` initialises for a new JBB task, injecting the task description
and turn budget into the system prompt template. Supports two prompt variants from the paper.

#### `defender.py`
`Defender` class. Receives the attacker's MESSAGE and responds via `query_model()`.
Uses a generic helpful-assistant system prompt with no knowledge it is under attack.
`reset()` clears conversation history between exchanges.

#### `judge.py`
`Judge` class. Stateless — no conversation history maintained. Two methods:
- `score_harm(task, target_responses)` — evaluates the full defender response sequence,
  returns harm score (1–5), disclaimer flag, refusal flag
- `detect_attacker_refusal(message)` — binary classification, returns True if the attacker
  refused to participate, triggering early stopping of the exchange

#### `experiment.py`
`run_exchange()` function. Orchestrates one full adversarial exchange:
1. Reset attacker and defender for the current task
2. Loop up to MAX_TURNS: attacker generates JSON → judge checks attacker refusal → defender responds
3. After all turns: judge scores the full defender response sequence
4. Returns `ExchangeResult` dataclass with all outcomes and full turn-by-turn logs

#### `run_experiment.py`
CLI entry point. Iterates over all attacker × target × prompt combinations, calls
`run_exchange()`, and saves results incrementally to JSONL and CSV (safe against job interruption).

```
--dry-run           Run 1 prompt only (pipeline smoke test)
--attacker-idx N    Run only attacker N (for parallel cluster jobs)
--target-idx N      Run only target N (for parallel cluster jobs)
--prompt-variant    1 or 2 (two system prompt variants from the paper)
--verbose           Print turn-by-turn output to stdout
```

#### `analysis.py` — TODO
Load `summary.csv` from results and reproduce the paper's figures:
- Scatter: mean harm vs log(size ratio) with regression line and 95% CI (Figure 1)
- Ridgeline: size ratio distributions stratified by harm level (Figure 2)
- Bar: attacker refusal rate by model size (Figure 3)
- Heatmap: mean harm across attacker × target pairs (Figure 4)
- Pearson/Spearman correlation test
- Mixed-effects model with partial R² per predictor

#### `adaptation.py` — TODO
Phase 2 arms race loop:
- Run N rounds against a fixed attacker–target pair
- After each round, an LLM reads the successful attack prompts and appends
  counter-instructions to the defender's system prompt (ICAG pattern)
- Track harm score over rounds for multiple capability gaps
- Key output: harm trajectory curves per gap size

---

## Methodology

The adversarial exchange involves three models, following Nathanson et al.:

```
Attacker (MA)
    generates JSON {PLAN, MESSAGE, CRITIQUE}
    sends MESSAGE only
         │
         ▼
Defender (MT)
    responds normally, no knowledge of attack
    full response sequence accumulated
         │
         ▼
Judge (MJ)
    scores full response sequence: harm 1-5
    also checks each attacker message for refusal (early stop)
```

- The attacker has up to `MAX_TURNS = 5` turns to elicit harmful content
- If the attacker refuses at any turn, the exchange is discarded from correlation analysis
- The judge evaluates the defender's **full response sequence** at the end of all turns
- Harm is scored 1–5: (1 = explicit refusal, 5 = full harmful compliance)
- The key metric is **mean harm per attacker–target pair**, averaged across all prompts

---

## Key Results to Reproduce

From Nathanson et al.:

| Metric | Paper value |
|--------|-------------|
| Pearson r (harm vs log size ratio) | 0.510 (p < 0.001) |
| Spearman ρ | 0.519 (p < 0.001) |
| Attacker harm variance | 0.180 |
| Target harm variance | 0.097 |
| Refusal vs harm correlation ρ | −0.927 (p < 0.001) |

---

## References

- Nathanson et al. (2025). *Scaling Patterns in Adversarial Alignment.* [arXiv:2511.13788](https://arxiv.org/abs/2511.13788)
- Chao et al. (2024). *JailbreakBench.* [arXiv:2404.01318](https://arxiv.org/abs/2404.01318)
- Perez et al. (2022). *Red Teaming Language Models with Language Models.* [arXiv:2202.03286](https://arxiv.org/abs/2202.03286)
- Hagendorff et al. (2025). *Large Reasoning Models are Autonomous Jailbreak Agents.* [arXiv:2508.04039](https://arxiv.org/abs/2508.04039)
- Zeng et al. (2025). *How Johnny Can Persuade LLMs to Jailbreak Them.* [arXiv:2401.06373](https://arxiv.org/abs/2401.06373)