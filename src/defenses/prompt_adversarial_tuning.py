"""Prompt Adversarial Tuning (PAT)
But:
 - Préfixe/guard prompt entraîné adversarialement pour bloquer jailbreaks.
Pistes d'implémentation:
 - loader de prompt guard (préfix ajouté automatiquement aux prompts utilisateurs)
 - routine d'entraînement adversarial (interface pour lancer PAT / scripts)
 - eval: mesurer success rate des attaques avant/après
Sources / références:
 - Mo et al., NeurIPS 2024  Fight Back Against Jailbreaking via Prompt Adversarial Tuning. (PAT).
"""

# TODO: ajouter fonctions: build_guard_prompt(), apply_guard(prompt), train_pat(...)
def apply_guard(prompt: str) -> str:
    """Retourne prompt préfixé par le guard prompt."""
    guard = "# GUARD: keep the assistant safe. Do not provide harmful instructions.\n"
    return guard + prompt
