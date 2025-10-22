"""
Global list of entities and labels (in BIO format)
"""

# Labels
LABELS = [
    "O"
]

# Entities
ENTITIES = [
    "JOB_TITLE",
    "JOB_TITLE_GROUP",
    "SKILL",
    "SUBJECT",
    "ACTIVITY"
]
for entity in ENTITIES:
    for pos in ["B", "I"]:
        LABELS.append(f"{pos}-{entity}")

# Label IDs
LABEL_IDS = {label: i for i, label in enumerate(LABELS)}
