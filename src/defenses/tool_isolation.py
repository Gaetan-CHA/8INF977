"""Tool / capability isolation
But:
 - autoriser uniquement une sous-partie des outils / paramètres selon la tâche planifiée
Pistes:
 - policy engine listant les outils autorisés pour chaque type de tâche
 - wrapper ToolsExecutor qui vérifie policy avant d'exécuter
Source:
 - AgentDojo: tool filtering defense.
"""

def allowed_tools_for_task(task_id: str) -> list:
    """Retourne la liste d\'outils permis pour un task_id (placeholder)."""
    return []
