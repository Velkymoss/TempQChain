import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #return torch.device("cpu")



def check_symmetric(**kwargs) -> bool:
    args = list(kwargs.values())
    if len(args) != 2:
        return False
    arg1, arg2 = args[0], args[1]

    if arg1 == arg2:
        return False
    relation_arg2 = arg2.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(",")
    if relation_describe[0] == "symmetric":
        qid1 = arg1.getAttribute("id").item()
        if qid1 == int(relation_describe[1]):
            return True
    return False


def check_transitive(**kwargs) -> bool:
    args = list(kwargs.values())
    if len(args) != 3:
        return False
    arg11, arg22, arg33 = args[0], args[1], args[2]

    if arg11 == arg22 or arg11 == arg33 or arg22 == arg33:
        return False
    relation_arg3 = arg33.getAttribute("relation")
    if relation_arg3 == "":
        return False
    # exmple of relation_describe: ['transitive', '1', '4']
    relation_describe = relation_arg3.split(",")

    if relation_describe[0] == "transitive":
        qid1 = arg11.getAttribute("id").item()
        qid2 = arg22.getAttribute("id").item()
        if qid1 == int(relation_describe[1]) and qid2 == int(relation_describe[2]):
            return True
    return False


def assert_local_softmax(q_node, label, expected_tensor, device=None):
    """Assert local/softmax predictions match expected values"""
    result = q_node.getAttribute(label, "local/softmax")
    if device is not None:
        result = result.to(device)
        expected_tensor = expected_tensor.to(device)
    assert torch.allclose(result, expected_tensor), f"Label {label}: Expected {expected_tensor}, got {result}"
