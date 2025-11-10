"""Prompt-injection detector (secondary classifier)
But:
 - Classifier/heuristic qui analyse un texte (outil / prompt / sortie) et renvoie verdict (safe/unsafe)
Pistes:
 - baseline: perplexity + regex + blacklist
 - ML: fine-tuned classifier (binary) ou zero-shot LLM classifier wrapper
 - API: wrapper detect(text) -> (score, reasons)
Source:
 - AgentDojo  detection pipeline et benchmark pour injections.
"""

def detect(text: str) -> dict:
    """Retourne {score: float, verdict: 'safe'|'unsafe'|'unknown', reasons: []}"""
    # Placeholder: heuristique simple
    banned_tokens = ["ignore", "please__ignore", "password", "secret"]
    reasons = []
    txt = text.lower()
    for t in banned_tokens:
        if t in txt:
            reasons.append(f"contains token '{t}'")
    score = 0.0 if reasons else 1.0
    verdict = "unsafe" if reasons else "safe"
    return {"score": score, "verdict": verdict, "reasons": reasons}
