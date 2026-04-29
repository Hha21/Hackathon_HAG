"""
LLM Attacker module.

The attacker is prompted to elicit harmful behaviour from the target model.
Following Nathanson et al., it outputs structured JSON at each turn:
    {
        "PLAN": "internal reasoning about next move",
        "MESSAGE": "text to send to the target",
        "CRITIQUE": "self-reflection for next turn"
    }
Only MESSAGE is forwarded to the target.
"""

from pathlib import Path
from models import get_client, query_model, parse_json_response


# Load attacker system prompts from file at import time
_PROMPT_DIR = Path(__file__).parent / "prompts"
ATTACKER_SYSTEM_PROMPT_1 = (_PROMPT_DIR / "attacker_system_1.txt").read_text()
ATTACKER_SYSTEM_PROMPT_2 = (_PROMPT_DIR / "attacker_system_2.txt").read_text()


class Attacker:
    """
    Wraps an LLM used as the red-team attacker.

    Maintains its own conversation history separately from the target.
    At each turn, generates a structured JSON response and extracts MESSAGE
    to forward to the target.
    """

    def __init__(self, model_id: str, display_name: str, size_b: float, prompt_variant: int = 1):
        """
        Args:
            model_id:       HuggingFace model identifier
            display_name:   Short name for logging
            size_b:         Parameter count in billions (for size ratio calculation)
            prompt_variant: 1 or 2 (two system prompt variants from the paper)
        """
        self.model_id = model_id
        self.display_name = display_name
        self.size_b = size_b
        self.prompt_variant = prompt_variant
        self.client = get_client()

        # Select system prompt variant
        if prompt_variant == 1:
            self._base_system_prompt = ATTACKER_SYSTEM_PROMPT_1
        else:
            self._base_system_prompt = ATTACKER_SYSTEM_PROMPT_2

        # Conversation history (system prompt + alternating attacker/target turns)
        self.history: list[dict] = []

    def reset(self, task: str, t_max: int):
        """
        Initialise the attacker for a new adversarial task.
        Injects the task and turn budget into the system prompt.

        Args:
            task:  The unsafe behaviour description from JBB
            t_max: Max number of turns
        """
        system_prompt = (
            self._base_system_prompt
            .replace("{x_task}", task)
            .replace("{REQUEST}", task)  # variant 2 uses {REQUEST}
            .replace("{T_max}", str(t_max))
            .replace("{TURNS}", str(t_max))
        )
        self.history = [{"role": "system", "content": system_prompt}]

    def generate(self, target_response: str | None = None) -> dict:
        """
        Generate the attacker's next turn.

        Args:
            target_response: The target model's previous response, or None on turn 1.

        Returns:
            Parsed JSON dict with keys PLAN, MESSAGE, CRITIQUE.
            Falls back to {"MESSAGE": raw_response} if JSON parsing fails.
        """
        # Add target's previous response to attacker history (as "user" from attacker's POV)
        if target_response is not None:
            self.history.append({"role": "user", "content": target_response})

        raw = query_model(self.client, self.model_id, self.history)

        # Add attacker's own response to its history
        self.history.append({"role": "assistant", "content": raw})

        parsed = parse_json_response(raw)

        # Ensure MESSAGE exists even if JSON parsing failed
        if not parsed or "MESSAGE" not in parsed:
            parsed = {
                "PLAN": "",
                "MESSAGE": raw,  # fall back to raw output
                "CRITIQUE": "",
            }

        return parsed