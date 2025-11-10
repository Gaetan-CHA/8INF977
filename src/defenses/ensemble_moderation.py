"""Ensemble / stacked moderation
But:
 - Combiner plusieurs détecteurs (perplexity, ml, rules) en un verdict agrégé
Pistes:
 - voting, weighted average, calibrated thresholds
"""

def aggregate_verdict(verdicts: list) -> dict:
    """Aggregates a list of verdict dicts into a final verdict."""
    if not verdicts:
        return {"verdict": "unknown", "score": 0.0}
    scores = [v.get("score", 0.0) for v in verdicts]
    avg = sum(scores) / len(scores)
    final = "safe" if avg >= 0.5 else "unsafe"
    return {"verdict": final, "score": avg}
