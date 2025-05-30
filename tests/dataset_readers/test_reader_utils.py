import pytest
from domino_readers.utils import label_fr_to_int
from domino_readers.reader_classes import TrainReader

# def test_yn_question_type():
#     """Test create_key with Yes/No question type"""
#     result = create_key("obj1", "obj2", "ABOVE", "YN")
#     assert result == "obj1:obj2:ABOVE"

# def test_fr_question_type():
#     """Test create_key with Free Response question type"""
#     result = create_key("obj1", "obj2", "ABOVE", "FR")
#     assert result == "obj1:obj2"

# def test_special_characters():
#     """Test create_key with object names containing special characters"""
#     result = create_key("obj:1", "obj:2", "NEAR", "YN")
#     assert result == "obj:1:obj:2:NEAR"

# def test_empty_strings():
#     """Test create_key with empty strings"""
#     result = create_key("", "", "RIGHT", "YN")
#     assert result == "::RIGHT"

# @pytest.mark.parametrize("relation", [
#     "LEFT", "RIGHT", "ABOVE", "BELOW", "BEHIND", "FRONT", "NEAR", "FAR"
# ])
# def test_different_relations(relation):
#     """Test create_key with different spatial relations"""
#     result = create_key("obj1", "obj2", relation, "YN")
#     assert result == f"obj1:obj2:{relation}" 

# @pytest.fixture
# def obj_info():
#     return {"obj1": {"full_name": "red cube"},
#             "obj2": {"full_name": "blue sphere"}}

# def test_create_simple_question_yn_relation_str(monkeypatch, obj_info):
#     # Use a relation whose VOCABULARY entry is a string
#     question = create_simple_question("obj1", "obj2", "DC", obj_info, "YN")
#     assert question == "Is red cube disconnected from blue sphere?"

# def test_create_simple_question_fr_first(monkeypatch, obj_info):
#     # Force random.random() < 0.5 to get question_fr1
#     monkeypatch.setattr("random.random", lambda: 0.1)
#     question = create_simple_question("obj1", "obj2", "LEFT", obj_info, "FR")
#     assert question == "Where is red cube relative to the blue sphere?"

# def test_create_simple_question_fr_second(monkeypatch, obj_info):
#     # Force random.random() >= 0.5 to get question_fr2
#     monkeypatch.setattr("random.random", lambda: 0.9)
#     question = create_simple_question("obj1", "obj2", "LEFT", obj_info, "FR")
#     print(question)
#     assert question == "What is the position of the red cube regarding blue sphere?"

# def test_create_simple_question_obj_names_with_spaces(monkeypatch):
#     obj_info = {
#         "obj1": {"full_name": "big green box"},
#         "obj2": {"full_name": "small yellow ball"}
#     }
#     monkeypatch.setattr("random.random", lambda: 0.1)
#     question = create_simple_question("obj1", "obj2", "RIGHT", obj_info, "FR")
#     assert question == "Where is big green box relative to the small yellow ball?"

# def test_create_simple_question_invalid_relation(monkeypatch, obj_info):
#     # Should raise KeyError if relation not in VOCABULARY
#     with pytest.raises(KeyError):
#         create_simple_question("obj1", "obj2", "INVALID", obj_info, "YN")

# def test_create_simple_question_missing_obj(monkeypatch, obj_info):
#     # Should raise KeyError if obj1 not in obj_info
#     with pytest.raises(KeyError):
#         create_simple_question("missing", "obj2", "LEFT", obj_info, "YN")

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