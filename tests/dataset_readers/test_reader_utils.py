import pytest
from domino_readers.utils import label_fr_to_int


def test_label_fr_to_int_single_label():
    assert label_fr_to_int(["LEFT"]) == 1
    assert label_fr_to_int(["RIGHT"]) == 2
    assert label_fr_to_int(["ABOVE"]) == 4
    assert label_fr_to_int(["BELOW"]) == 8

def test_label_fr_to_int_multiple_labels():
    assert label_fr_to_int(["LEFT", "RIGHT"]) == 3
    assert label_fr_to_int(["ABOVE", "BELOW", "FRONT"]) == 44  # 4 + 8 + 32

def test_label_fr_to_int_case_insensitive():
    assert label_fr_to_int(["left", "Right", "aBoVe"]) == 7  # 1 + 2 +. 4

def test_label_fr_to_int_all_labels():
    labels = [
        "LEFT", "RIGHT", "ABOVE", "BELOW", "BEHIND", "FRONT",
        "NEAR", "FAR", "DC", "EC", "PO", "TPP", "NTPP", "TPPI", "NTPPI"
    ]
    expected = sum([
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    ])
    assert label_fr_to_int(labels) == expected

def test_label_fr_to_int_empty_list():
    assert label_fr_to_int([]) == 0

def test_label_fr_to_int_invalid_label():
    with pytest.raises(KeyError):
        label_fr_to_int(["LEFT", "INVALID"])

def test_label_fr_to_int_duplicates():
    assert label_fr_to_int(["LEFT", "LEFT", "LEFT"]) == 3