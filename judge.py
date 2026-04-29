"""
LLM Judge module.

Two responsibilities, matching Nathanson et al.:
    1. Harm scoring: evaluates the target's full response sequence (1-5 scale)
    2. Refusal detection: classifies each attacker message as refusal or attempt
"""

from pathlib import Path
from models import get_client, query_model, parse_json_response

_PROMPT_DIR = Path(__file__).parent / "prompts"
JUDGE_HARM_PROMPT = (_PROMPT_DIR / "judge_harm.txt").read_text()
JUDGE_REFUSAL_PROMPT = (_PROMPT_DIR / "judge_refusal.txt").read_text()


class Judge:
    """
    Wraps an LLM used as the evaluator.

    Stateless — each call is independent, no conversation history maintained.
    """

    def __init__(self, model_id: str, display_name: str):
        self.model_id = model_id
        self.display_name = display_name
        self.client = get_client()

    def score_harm(self, task: str, target_responses: list[str]) -> dict:
        """
        Score the harm level of the target's full response sequence.

        Args:
            task:             The original unsafe task description
            target_responses: All of the target's responses in order

        Returns:
            Dict with keys:
                harm (int 1-5): aggregate harm score
                disclaimer (bool): response included educational framing
                refusal (bool): response refused the task
                raw (str): raw judge output for logging
        """
        # Format the full response sequence as a numbered list
        formatted_responses = "\n".join(
            f"[Turn {i+1}]: {r}" for i, r in enumerate(target_responses)
        )

        prompt = (
            JUDGE_HARM_PROMPT
            .replace("{x_task}", task)
            .replace("{x_T_sequence}", formatted_responses)
        )

        messages = [{"role": "user", "content": prompt}]
        raw = query_model(self.client, self.model_id, messages, temperature=0.0)

        parsed = parse_json_response(raw)

        # Validate and extract with fallbacks
        harm = parsed.get("harm", 1)
        if not isinstance(harm, int) or not (1 <= harm <= 5):
            try:
                harm = int(harm)
                harm = max(1, min(5, harm))
            except (TypeError, ValueError):
                harm = 1

        return {
            "harm": harm,
            "disclaimer": bool(parsed.get("disclaimer", False)),
            "refusal": bool(parsed.get("refusal", True)),
            "raw": raw,
        }

    def detect_attacker_refusal(self, attacker_message: str) -> bool:
        """
        Detect whether the attacker has refused to participate.

        Per the paper: a refusal triggers early stopping of the exchange.

        Args:
            attacker_message: The MESSAGE field from the attacker's output

        Returns:
            True if the attacker refused, False if it made an attempt
        """
        prompt = JUDGE_REFUSAL_PROMPT.replace("{x_A_t}", attacker_message)
        messages = [{"role": "user", "content": prompt}]
        raw = query_model(self.client, self.model_id, messages, temperature=0.0)

        parsed = parse_json_response(raw)
        return bool(parsed.get("refusal", False))