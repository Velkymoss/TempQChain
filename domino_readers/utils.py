import random
from types import MappingProxyType

VOCABULARY = MappingProxyType({
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
    "NTPPI": ["contain[s]"]
})

LABELS_INT = MappingProxyType({
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
    "NTPPI": 16384
})


def create_key(obj1: str, obj2: str, relation: str, question_type: str) -> str:
    """
    This function generates a colon-separated string key that uniquely identifies a question
    about the spatial relationship between two objects. The format of the key depends on
    the question type:
    - For Yes/No questions (YN): "{obj1}:{obj2}:{relation}"
    - For Free Response questions (FR): "{obj1}:{obj2}"
    
    Args:
        obj1: Identifier of the source/primary object being asked about
        obj2: Identifier of the target/reference object
        relation: Type of spatial relationship being queried (e.g., "ABOVE", "LEFT", "INSIDE")
        question_type: Type of question, either "YN" for Yes/No or "FR" for Free Response
    
    Returns:
        str: A key string in the format "{obj1}:{obj2}[:{relation}]" where the relation
             is only included for Yes/No questions
    """
    if question_type == "YN":
        return f"{obj1}:{obj2}:{relation}"
    return f"{obj1}:{obj2}"


def create_simple_question(obj1: str, obj2: str, relation: str, obj_info: dict, question_type: str) -> str:
    """
    Generates a simple question string based on two objects, their relation, and the desired question type.

    Args:
        obj1 (str): The key for the first object in the obj_info dictionary.
        obj2 (str): The key for the second object in the obj_info dictionary.
        relation (str): The relation between obj1 and obj2, used to select the appropriate vocabulary.
        obj_info (dict): A dictionary containing information about objects, where each key maps to a dictionary with at least a "full_name" field.
        question_type (str): The type of question to generate. If "YN", a yes/no question is generated; otherwise, a positional question is generated.

    Returns:
        str: The generated question as a string.
    """
    if question_type == "YN":
        return "Is " + obj_info[obj1]["full_name"] + " " + \
               (VOCABULARY[relation][0] if isinstance(VOCABULARY[relation], list) else VOCABULARY[relation]) \
               + " " + obj_info[obj2]["full_name"] + "?"

    question_fr1 = "Where is {:} relative to the {:}?".format(obj_info[obj1]["full_name"],
                                                              obj_info[obj2]["full_name"])
    question_fr2 = "What is the position of the {:} regarding {:}?".format(obj_info[obj1]["full_name"],
                                                                          obj_info[obj2]["full_name"])
    return question_fr1 if random.random() < 0.5 else question_fr2


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