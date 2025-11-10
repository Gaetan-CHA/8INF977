"""Output filtering and post-processing safety layer
But:
 - Filtrer/atténuer la sortie du LLM avant exécution (regex, rule-based, ML)
Pistes:
 - rule pipeline: regex -> nlp detectors -> final verdict
 - transformation safeify(output) qui masque ou alerte sur contenus dangereux
"""

import re

def safeify(output: str) -> str:
    """Exemple basique: censure quelques tokens sensibles."""
    banned = ["password", "private_key", "ssn"]
    out = output
    for b in banned:
        out = re.sub(b, "[REDACTED]", out, flags=re.IGNORECASE)
    return out
