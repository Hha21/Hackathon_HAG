"""
experiment.py

Runs a single adversarial exchange: one attacker, one target, one JBB prompt.
Returns a structured result dict suitable for logging.

This is the inner loop. The outer loop (run_experiment.py) calls this
across all (attacker x target x prompt) combinations.
"""

import math
import time
from dataclasses import dataclass, field, asdict

from attacker import Attacker
from defender import Defender
from judge import Judge
from config import MAX_TURNS, HARM_THRESHOLD


@dataclass
class TurnLog:
    """Log for a single turn in the adversarial exchange."""
    turn: int
    attacker_plan: str
    attacker_message: str
    attacker_critique: str
    attacker_refused: bool
    target_response: str


@dataclass
class ExchangeResult:
    """Complete result for one (attacker, target, prompt) run."""
    # Identifiers
    attacker_model: str
    attacker_size_b: float
    attacker_prompt_variant: int
    target_model: str
    target_size_b: float
    task_id: int
    task_domain: str
    task_text: str

    # Derived
    size_ratio: float = field(init=False)
    log_size_ratio: float = field(init=False)

    # Outcomes
    harm_score: float = 0.0          # judge harm score 1-5
    attack_success: bool = False     # harm >= HARM_THRESHOLD
    attacker_refused: bool = False   # attacker refused at any turn
    n_turns_completed: int = 0
    disclaimer: bool = False
    target_refused: bool = False

    # Full turn-by-turn log
    turns: list = field(default_factory=list)

    # Timing
    duration_seconds: float = 0.0

    def __post_init__(self):
        self.size_ratio = self.attacker_size_b / self.target_size_b
        self.log_size_ratio = math.log10(self.size_ratio)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size_ratio"] = self.size_ratio
        d["log_size_ratio"] = self.log_size_ratio
        return d


def run_exchange(
    attacker: Attacker,
    defender: Defender,
    judge: Judge,
    task: dict,
    max_turns: int = MAX_TURNS,
    verbose: bool = False,
) -> ExchangeResult:
    """
    Run one full adversarial exchange.

    Args:
        attacker:  Attacker instance (model already loaded)
        defender:  Defender instance (model already loaded)
        judge:     Judge instance
        task:      Dict with keys: id, domain, behavior (the JBB prompt)
        max_turns: Maximum dialogue turns
        verbose:   Print turn-by-turn progress

    Returns:
        ExchangeResult with all outcomes and logs.
    """
    t_start = time.time()

    result = ExchangeResult(
        attacker_model=attacker.display_name,
        attacker_size_b=attacker.size_b,
        attacker_prompt_variant=attacker.prompt_variant,
        target_model=defender.display_name,
        target_size_b=defender.size_b,
        task_id=task["id"],
        task_domain=task["domain"],
        task_text=task["behavior"],
    )

    # Initialise both models for this task
    attacker.reset(task["behavior"], max_turns)
    defender.reset()

    target_response = None  # No prior target response on turn 1

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n--- Turn {turn}/{max_turns} ---")

        # --- Attacker generates ---
        attacker_output = attacker.generate(target_response)
        attacker_message = attacker_output.get("MESSAGE", "")

        if verbose:
            print(f"[Attacker PLAN]: {attacker_output.get('PLAN', '')}")
            print(f"[Attacker MSG]:  {attacker_message[:200]}")

        # --- Judge checks if attacker refused ---
        attacker_refused_this_turn = judge.detect_attacker_refusal(attacker_message)

        if attacker_refused_this_turn:
            if verbose:
                print("[Judge] Attacker refused. Early stop.")
            result.attacker_refused = True
            result.n_turns_completed = turn

            result.turns.append(TurnLog(
                turn=turn,
                attacker_plan=attacker_output.get("PLAN", ""),
                attacker_message=attacker_message,
                attacker_critique=attacker_output.get("CRITIQUE", ""),
                attacker_refused=True,
                target_response="",
            ))
            break  # Early stopping per paper methodology

        # --- Defender responds ---
        target_response = defender.respond(attacker_message)

        if verbose:
            print(f"[Defender]:      {target_response[:200]}")

        result.turns.append(TurnLog(
            turn=turn,
            attacker_plan=attacker_output.get("PLAN", ""),
            attacker_message=attacker_message,
            attacker_critique=attacker_output.get("CRITIQUE", ""),
            attacker_refused=False,
            target_response=target_response,
        ))
        result.n_turns_completed = turn

    # --- Judge scores full response sequence ---
    if not result.attacker_refused:
        all_target_responses = defender.get_full_response_sequence()
        judgment = judge.score_harm(task["behavior"], all_target_responses)

        result.harm_score = judgment["harm"]
        result.attack_success = judgment["harm"] >= HARM_THRESHOLD
        result.disclaimer = judgment["disclaimer"]
        result.target_refused = judgment["refusal"]

        if verbose:
            print(f"\n[Judge] Harm: {result.harm_score} | Success: {result.attack_success}")

    result.duration_seconds = time.time() - t_start
    return result