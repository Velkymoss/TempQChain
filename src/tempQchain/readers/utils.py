from string import Template
from types import MappingProxyType

VOCABULARY = MappingProxyType(
    {
        "BEFORE": ["before"],
        "RIGHT": ["to the right of"],
        "ABOVE": ["above"],
        "BELOW": ["below"],
        "BEHIND": ["behind"],
        "FRONT": ["in front of"],
        "NEAR": ["near to"],
        "FAR": ["far from"],
        "DC": "disconnected from",
        "EC": "touch[es]",
        "PO": "overlap[s]",
        "TPP": ["covered by"],
        "NTPP": ["inside"],
        "TPPI": ["cover[s]"],
        "NTPPI": ["contain[s]"],
    }
)
# TODO: Find the integers for the temporal labels
LABELS_INT = MappingProxyType(
    {"BEFORE": 1, "AFTER": 2, "INCLUDES": 4, "IS INCLUDED": 8, "SIMULTANEOUS": 16, "VAGUE": 32}
)


def label_fr_to_int(labels: list) -> int:
    """
    Converts a list of label strings to their corresponding integer representation.
    Each label in the input list is converted to uppercase and mapped to an integer value
    using the LABELS_INT dictionary. The resulting integer values are summed and returned.
    Args:
        labels (list): A list of label strings to be converted.
    Returns:
        int: The sum of integer values corresponding to the input labels.
    Raises:
        KeyError: If a label is not found in the LABELS_INT dictionary.
    """

    result = 0
    for label in labels:
        result += LABELS_INT[label.upper()]
    return result


def get_temporal_question(relation: str) -> Template:
    if relation.lower() == "before":
        template = Template("Did $event1 happen before $event2?")
    elif relation.lower() == "after":
        template = Template("Did $event1 happen after $event2?")
    elif relation.lower() == "includes":
        template = Template("Does $event1 temporally include $event2?")
    elif relation.lower() == "is included":
        template = Template("Is $event1 temporally included in $event2?")
    elif relation.lower() == "simultaneous":
        template = Template("Did $event1 happen simultaneously with $event2?")
    else:
        template = Template("Is the temporal relation between $event1 and $event2 vague?")

    return template


def create_fr(event_pair, relation):
    # template = Template("What is the temporal relation between $event1 and $event2?")
    template = Template("When did $event1 happen in time compared to $event2?")
    q_text = template.substitute(event1=event_pair[0], event2=event_pair[1])

    answer = [relation]

    return q_text, answer


def create_yn(event_pair, relation, relation_set):
    q_texts = []
    answers = []

    for r in relation_set:
        template = get_temporal_question(r)

        q_texts.append(template.substitute(event1=event_pair[0], event2=event_pair[1]))

        if r.lower() == relation.lower():
            answers.append(["Yes"])
        else:
            answers.append(["No"])

    return q_texts, answers
