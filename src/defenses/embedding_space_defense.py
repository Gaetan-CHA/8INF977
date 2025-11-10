"""Defenses against embedding-space / soft-prompt attacks
But:
 - détecter perturbations d\'embeddings et rétablir robustesse
Pistes:
 - monitor embedding drift, anomaly detection in embedding space
 - projection/clipping / input denoising
 - normalisation et clipping des embeddings, détection d\'anomalies via distance au centroid
Source:
 - Soft-Prompt Threats (NeurIPS 2024)
"""

from typing import Sequence
import numpy as np

def detect_embedding_anomaly(embedding: Sequence[float], reference_centroid: Sequence[float], threshold: float = 5.0) -> bool:
    """
    Détecte si 'embedding' est anormal par rapport au centroid de référence.
    Simple implémentation: distance Euclidienne > threshold -> anomalie.
    """
    try:
        e = np.array(embedding, dtype=float)
        c = np.array(reference_centroid, dtype=float)
        dist = np.linalg.norm(e - c)
        return float(dist) > float(threshold)
    except Exception:
        # En cas d'erreur -> considérer comme anomalie conservatrice
        return True
