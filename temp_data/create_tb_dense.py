import json
import os
from statistics import mean

import pandas as pd
from bs4 import BeautifulSoup
from data_utils import (
    build_data,
    check_trans_for_pairs,
    create_chain,
    create_fr,
    create_object_info,
    create_story_triplets,
    create_yn,
    extract_all_entities,
    extract_story,
    read_txt,
    save_json,
)

from logger import get_logger

logger = get_logger(__name__)

tb_dense_lines = read_txt(os.path.join("tb_dense", "TimebankDense.full.txt"))
tb_dense_df = pd.DataFrame(tb_dense_lines, columns=["doc_id", "event1_id", "event2_id", "relation"])

tb_dense_docs = list(tb_dense_df.doc_id.unique())
logger.info(f"There are {len(tb_dense_docs)} documents with {len(tb_dense_df)} relations in total.")

rel = {"s": "simultaneous", "i": "includes", "a": "after", "v": "vague", "ii": "is included", "b": "before"}
tb_dense_df["relation"].replace(rel, inplace=True)

# Get pairs and relations of each document
doc_pair_relations = []
for doc_id in tb_dense_df.doc_id.unique():
    doc_tlinks = tb_dense_df.loc[tb_dense_df["doc_id"] == doc_id]
    doc_pairs = list(zip(doc_tlinks.event1_id.to_list(), doc_tlinks.event2_id.to_list()))
    doc_relations = doc_tlinks.relation.to_list()
    doc_pair_relations.append(dict(zip(doc_pairs, doc_tlinks.relation.to_list())))

num_pairs = []
for p in doc_pair_relations:
    num_pairs.append(len(p))
logger.info(f"The number of pairs in a report on average is {mean(num_pairs)}")

# Get trans triples
trans_counts = []
trans_pairs = []
num_trans = []
for pr in doc_pair_relations:
    report_pairs = list(pr.keys())
    t_count, t_pairs = check_trans_for_pairs(report_pairs)
    trans_counts.append(t_count)
    trans_pairs.append(t_pairs)
    for triple in t_pairs:
        # if pr[triple[0]] != "vague" and pr[triple[1]] != "vague" and pr[triple[2]] != "vague":
        num_trans.append(1)
logger.info(f"Transitivity appears on average: {mean(trans_counts)}")

logger.info(f"There are {sum(num_trans)} transitivity triples")

doc_story_triplets = create_story_triplets(doc_pair_relations)

if len(doc_story_triplets[0]) != len(doc_pair_relations[0]):
    raise ValueError("The number of pairs in the story triplets does not match the number of pairs in the relations.")

doc_objects_info = create_object_info(tb_dense_df)

inverse = {
    "before": "after",
    "after": "before",
    "includes": "is included",
    "is included": "includes",
    "overlap": "overlap",
    "simultaneous": "simultaneous",
    "vague": "vague",
}

trans_triples = trans_pairs

doc_chains = create_chain(doc_pair_relations, trans_pairs, inverse)

relation_set = list(tb_dense_df.relation.unique())

doc_questions = []
for doc_index, doc_id in enumerate(tb_dense_df.doc_id.unique()):
    # Get each document's tlinks
    doc_tlinks = tb_dense_df.loc[tb_dense_df["doc_id"] == doc_id]

    questions = []

    # Create the question counter
    q_id = 0
    for index, row in doc_tlinks.iterrows():
        # Get the pair of events
        query = (row["event1_id"], row["event2_id"])

        # Add YN question (one for each relation)
        yn_questions, yn_answers = create_yn(query, row["relation"], relation_set)

        for i, yn_question in enumerate(yn_questions):
            question_info = {
                "num_facts": doc_chains[doc_index][query]["num_facts"],
                "reasoning_steps": doc_chains[doc_index][query]["reasoning_steps"],
                "asked_relation": relation_set[i],
                "all_relations": [row["relation"]],
                "target_relation": [row["relation"]],
                "chain": doc_chains[doc_index][query]["chain"],
                "goal_chain": doc_chains[doc_index][query]["goal_chain"],
            }

            questions.append(
                {
                    "q_id": q_id,
                    "q_type": "YN",
                    "query": query,
                    "question_info": question_info,
                    "question": yn_question,
                    "answer": yn_answers[i],
                    "candidate_answers": ["Yes", "No"],
                }
            )
            q_id += 1

        # Add the FR question
        question, answer = create_fr(query, row["relation"])
        question_info = {
            "num_facts": doc_chains[doc_index][query]["num_facts"],
            "reasoning_steps": doc_chains[doc_index][query]["reasoning_steps"],
            "asked_relation": [row["relation"]],
            "all_relations": [row["relation"]],
            "target_relation": [row["relation"]],
            "chain": doc_chains[doc_index][query]["chain"],
            "goal_chain": doc_chains[doc_index][query]["goal_chain"],
        }

        questions.append(
            {
                "q_id": q_id,
                "q_type": "FR",
                "query": query,
                "question_info": question_info,
                "question": question,
                "answer": answer,
                "candidate_answers": relation_set,
            }
        )
        q_id += 1

    doc_questions.append(questions)

data = build_data(tb_dense_docs, doc_story_triplets, doc_questions, doc_objects_info)
save_json("../data/", "tb_dense.json", data)

"""
This second part of the script reads the generated file and add the story text and event names to each datapoint.
"""

with open("../data/tb_dense.json", "r") as file:
    data = json.load(file)


story_path = "../data/timebank_1_2/data/extra/"


for datapoint in data["data"]:
    # open story text
    filepath = os.path.join(story_path, datapoint["identifier"] + ".tml")
    try:
        with open(filepath, "r") as file:
            content = file.read()
    except FileNotFoundError:
        raise (f"File not found: {filepath}")
    soup = BeautifulSoup(content, "xml")

    # add story text do dataset
    story_text = extract_story(soup)
    datapoint["story"] = story_text if story_text else ""

    # add event names to dataset
    story_events = extract_all_entities(soup)
    datapoint["event_names"] = story_events

# write to file
output_path = "../data/tb_dense.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as file:
    json.dump(data, file, indent=4)
