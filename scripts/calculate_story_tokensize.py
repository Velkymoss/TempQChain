import json
import os

from transformers import BertTokenizer

DATA_DIR = "data/"


def main():
    file_names = [f"tb_dense_{mode}.json" for mode in ["train", "dev", "test"]]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for file in file_names:
        file_path = os.path.join(DATA_DIR, file)
        with open(file_path, "r") as f:
            file = json.load(f)
        dataset = file["data"]
        for story in dataset:
            identifier = story["identifier"]
            story_str = story["story"][0]
            encoded_input = tokenizer(story_str, return_tensors="pt")
            print(f"{identifier}: {len(encoded_input['input_ids'][0])}")


if __name__ == "__main__":
    main()
