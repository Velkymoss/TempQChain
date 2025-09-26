import os
import sys
from statistics import mean

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import (
    build_data,
    check_trans_for_pairs,
    create_chain,
    create_facts_info,
    create_fr,
    create_object_info,
    create_story_triplets,
    create_yn,
    get_clean_article,
    get_t0,
    parse_article,
    read_txt,
    save_json,
    save_rules,
)

from logger import get_logger

logger = get_logger(__name__)

trans_rules = {  # rule 1
    ("before", "before"): ["before"],
    ("after", "after"): ["after"],
    ("includes", "includes"): ["includes"],
    ("is included", "is included"): ["is included"],
    ("simultaneous", "simultaneous"): ["simultaneous"],
    # ("vague", "vague"): ["vague"],
    # rule 2
    ("before", "simultaneous"): ["before"],
    ("after", "simultaneous"): ["after"],
    ("includes", "simultaneous"): ["includes"],
    ("is included", "simultaneous"): ["is included"],
    # ("vague", "simultaneous"): ["vague"],
    # ("vague", "before"): ["vague"],
    # ("vague", "after"): ["vague"],
    # ("vague", "includes"): ["vague"],
    # ("vague", "is included"): ["vague"],
    # Rules for before
    ("before", "after"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    ("before", "includes"): ["before", "includes", "vague"],
    ("before", "is included"): ["before", "is included", "vague"],
    # ("before", "vague"): ["before", "includes", "is included", "vague"],
    # ("before", "vague"): ["vague"],
    # Rules for after
    ("after", "before"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    ("after", "includes"): ["after", "includes", "vague"],
    ("after", "is included"): ["after", "is included", "vague"],
    # ("after", "vague"): ["after", "includes", "is included", "vague"],
    # ("after", "vague"): ["vague"],
    # Rules for includes
    ("includes", "before"): ["before", "includes", "vague"],
    ("includes", "after"): ["after", "includes", "vague"],
    ("includes", "is included"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    # ("includes", "vague"): ["before", "after", "includes", "vague"],
    # ("includes", "vague"): ["vague"],
    # Rules for is included
    ("is included", "before"): ["before", "is included", "vague"],
    ("is included", "after"): ["after", "is included", "vague"],
    ("is included", "includes"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    # ("is included", "vague"): ["before", "after", "is included", "vague"],
    # ("is included", "vague"): ["vague"],
    # Rules for simultaneous
    ("simultaneous", "before"): ["before"],
    ("simultaneous", "after"): ["after"],
    ("simultaneous", "includes"): ["includes"],
    ("simultaneous", "is included"): ["is included"],
    # ("simultaneous", "vague"): ["vague"],
}


def process_tb_dense(trans_rules: dict[tuple[str, str], list[str]] = trans_rules) -> list:
    # Load and preprocess data
    logger.info("Loading TB-Dense data...")
    path = "tb_dense"

    tb_dense_lines = read_txt(os.path.join(path, "TimebankDense.full.txt"))
    tb_dense_df = pd.DataFrame(tb_dense_lines, columns=["doc_id", "event1_id", "event2_id", "relation"])

    tb_dense_docs = list(tb_dense_df.doc_id.unique())
    logger.info(f"There are {len(tb_dense_docs)} documents with {len(tb_dense_df)} relations in total.")

    # Replace relation abbreviations with full names
    rel = {"s": "simultaneous", "i": "includes", "a": "after", "v": "vague", "ii": "is included", "b": "before"}
    tb_dense_df["relation"].replace(rel, inplace=True)

    # Get pairs and relations of each document
    logger.info("Processing document pairs and relations...")
    doc_pair_relations = []
    for doc_id in tb_dense_df.doc_id.unique():
        doc_tlinks = tb_dense_df.loc[tb_dense_df["doc_id"] == doc_id]
        doc_pairs = list(zip(doc_tlinks.event1_id.to_list(), doc_tlinks.event2_id.to_list()))
        doc_relations = doc_tlinks.relation.to_list()
        doc_pair_relations.append(dict(zip(doc_pairs, doc_tlinks.relation.to_list())))

    # Calculate statistics
    num_pairs = [len(p) for p in doc_pair_relations]
    logger.info(f"The number of pairs in a report on average is {mean(num_pairs)}")

    # Get transitivity triples
    logger.info("Finding transitivity triples...")
    trans_counts = []
    trans_pairs = []
    num_trans = []
    for pr in doc_pair_relations:
        report_pairs = list(pr.keys())
        t_count, t_pairs = check_trans_for_pairs(report_pairs)
        trans_counts.append(t_count)
        trans_pairs.append(t_pairs)
        for triple in t_pairs:
            num_trans.append(1)

    logger.info(f"Transitivity appears on average: {mean(trans_counts)}")
    logger.info(f"There are {sum(num_trans)} transitivity triples")

    # Construct story_triplets
    logger.info("Constructing story triplets...")
    doc_story_triplets = create_story_triplets(doc_pair_relations)

    # Construct objects_info
    logger.info("Constructing objects info...")
    doc_objects_info = create_object_info(tb_dense_df)

    # Construct chains
    logger.info("Constructing chains...")
    inverse = {
        "before": "after",
        "after": "before",
        "includes": "is included",
        "is included": "includes",
        # "overlap": "overlap",
        "simultaneous": "simultaneous",
        "vague": "vague",
    }

    doc_chains = create_chain(doc_pair_relations, trans_pairs, inverse)

    # Construct questions
    logger.info("Constructing questions...")
    relation_set = list(tb_dense_df.relation.unique())
    doc_questions = []

    for doc_index, doc_id in enumerate(tb_dense_df.doc_id.unique()):
        doc_tlinks = tb_dense_df.loc[tb_dense_df["doc_id"] == doc_id]
        questions = []
        q_id = 0

        for index, row in doc_tlinks.iterrows():
            query = (row["event1_id"], row["event2_id"])

            # Add YN questions (one for each relation)
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

    # Construct facts info
    logger.info("Constructing facts info...")

    doc_facts_info = create_facts_info(doc_questions, inverse, trans_rules)

    # Build final data structure
    logger.info("Building final data structure...")
    data = build_data(tb_dense_docs, doc_story_triplets, doc_questions, doc_objects_info, doc_facts_info)

    # Add story content from original articles
    logger.info("Adding story content from articles...")
    ARTICLE_PATH = "../data/timebank_1_2/data/extra/"

    for article in data:
        filename = article.get("identifier") + ".tml"
        filepath = os.path.join(ARTICLE_PATH, filename)
        article_soup = parse_article(filepath)
        t0_value = get_t0(article_soup)
        clean_article = get_clean_article(article_soup)
        article_header = f"Written on <EVENT> id=t0 >{t0_value}< </EVENT>"
        full_article = article_header + " " + clean_article
        article["story"] = [full_article]

    # Save to JSON
    logger.info("Saving data to JSON...")
    save_json(path, "tb_dense.json", data)

    # Save rules
    logger.info("Saving rules...")
    save_rules(inverse, "symmetry")

    logger.info("TB-Dense processing completed successfully!")
    return data


if __name__ == "__main__":
    process_tb_dense()
