from types import MappingProxyType

VOCABULARY = MappingProxyType(
    {
        "LEFT": ["to the left of"],
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

LABELS_INT = MappingProxyType(
    {
        "LEFT": 1,
        "RIGHT": 2,
        "ABOVE": 4,
        "BELOW": 8,
        "BEHIND": 16,
        "FRONT": 32,
        "NEAR": 64,
        "FAR": 128,
        "DC": 256,
        "EC": 512,
        "PO": 1024,
        "TPP": 2048,
        "NTPP": 4096,
        "TPPI": 8192,
        "NTPPI": 16384,
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
