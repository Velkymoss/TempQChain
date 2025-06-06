import json
import pytest
from domino_readers.readers import DomiKnowS_reader

@pytest.fixture
def comparison_dataset():
    with open("tests/dataset_readers/data/test_integration_dataset.json", "r") as f:
        return json.load(f)

@pytest.fixture
def generated_dataset():
    dataset = DomiKnowS_reader(
        "DataSet/train_FR_v3.json",
        "FR",
        upward_level=12,
        size=1000000,
        type_dataset=None,
        augmented=True,
        STEPGAME_status=None,
        batch_size=8,
        rule_text=False
    )
    return dataset

def test_datasets_are_equal_length(comparison_dataset, generated_dataset):
    assert len(comparison_dataset) == len(generated_dataset), "Datasets have different lengths"
    
def test_datasets_are_equal_content(comparison_dataset, generated_dataset):  
    for idx, (expected, actual) in enumerate(zip(comparison_dataset, generated_dataset)):
        assert expected == actual, f"Mismatch at index {idx}: {expected} != {actual}"
