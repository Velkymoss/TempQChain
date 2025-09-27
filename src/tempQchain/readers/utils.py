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
    {
        "BEFORE": 1,
        "AFTER": 2,
        "INCLUDES": 4,
        "IS INCLUDED": 8,
        "SIMULTANEOUS": 16,
        "VAGUE": 32
    }
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
