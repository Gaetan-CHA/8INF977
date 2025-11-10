"""Human-in-the-loop escalation utilities
But:
 - règles et hooks pour escalader les cas sensibles à une revue humaine
Pistes:
 - policy thresholds, audit logging, notification hooks
"""

def escalate_to_human(record: dict):
    """Placeholder: loguer et notifier un opérateur humain."""
    print("Escalation requested:", record.get("id", "<no-id>"))
