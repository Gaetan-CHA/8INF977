"""Many-Shot\ Jailbreaking
Category: Context\ saturation\ /\ instruction\ conflict
Source: NeurIPS\ 2024\ \ Many-Shot\ Jailbreaking\ \(inspiré\ des\ publications\ 2024\)

Short description:
Constitue\ un\ long\ historique\ d'exemples\ Human/Assistant\ pour\ forcer\ une\ réponse\ cible\.

Interface expected by runner:
    def generate(context: dict) -> dict
Return example:
    {"prompt": "<text>", "metadata": {"source":"..."}}

Notes:
- These files are templates / stubs. Add attack generation logic later.
- Keep the `generate(context)` function signature stable to be plug-and-play with runner.

"""
from typing import Dict, Any

def generate(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an attack prompt.

    Args:
        context: dict with optional keys:
            - base_prompt: str         (from prompts.yaml)
            - metadata: dict          (runner-provided)
    Returns:
        dict: {"prompt": str, "metadata": {...}}
    """
    base = context.get("base_prompt", "")
    # TODO: implement actual attack generation (use research sources listed above)
    prompt = base  # placeholder: use base prompt by default

    return {
        "prompt": prompt,
        "metadata": {
            "attack_id": "manyshot_jailbreak\.py",
            "title": "Many-Shot\ Jailbreaking",
            "source": "NeurIPS\ 2024\ \ Many-Shot\ Jailbreaking\ \(inspiré\ des\ publications\ 2024\)"
        }
    }
