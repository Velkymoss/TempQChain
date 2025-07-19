import os
import json

from string import Template


def read_txt(path_to_file):
    """

    :param path_to_file: Path to the txt dataset file
    :return: List with the lines of the txt file
    """
    with open(path_to_file, "r", encoding='utf-8-sig') as f:
        lines = f.readlines()

    tlinks = []
    for line in lines:
        tlinks.append(line.split())

    return tlinks


def check_trans_for_pairs(pairs):
    """

    :param pairs: A list of pairs
    :return: The count of transitive pairs found and a list with the triples of transitive pairs
    """
    trans_count = 0
    trans_pairs = []
    for p1 in pairs:
        p2 = []
        for p in pairs:
            if p[0] == p1[1]:
                p2.append(p)
        for second_pair in p2:
            if (p1[0], second_pair[1]) in pairs:
                trans_count += 1
                trans_pairs.append([p1, second_pair, (p1[0], second_pair[1])])
    return trans_count, trans_pairs


def create_story_triplets(doc_pair_relations):
    doc_story_triplets = []
    for doc in doc_pair_relations:
        story_triplets = {}
        for pair in doc:
            pair_key = str(pair)
            pair_value = [{"relation_type": doc[pair], "relation_property": ""}]
            story_triplets[pair_key] = pair_value
        doc_story_triplets.append(story_triplets)

    return doc_story_triplets


def create_object_info(data_df):
    doc_objects = []
    for doc_id in data_df.doc_id.unique():
        doc_tlinks = data_df.loc[data_df["doc_id"] == doc_id]
        # doc_events1 = dict(zip(doc_tlinks.event1_id.to_list(),
        #                        [{"name": text, "full_name": text} for text in doc_tlinks.event1_text.to_list()]))
        # doc_events2 = dict(zip(doc_tlinks.event2_id.to_list(),
        #                        [{"name": text, "full_name": text} for text in doc_tlinks.event2_text.to_list()]))

        doc_events1 = dict(zip(doc_tlinks.event1_id.to_list(), [{"name": "", "full_name": ""}]))
        doc_events2 = dict(zip(doc_tlinks.event2_id.to_list(), [{"name": "", "full_name": ""}]))

        doc_objects.append({**doc_events1, **doc_events2})

    return doc_objects


def get_temporal_question(relation):
    if relation.lower() == "before":
        template = Template("Did $event1 happen before $event2?")
    elif relation.lower() == "after":
        template = Template("Did $event1 happen after $event2?")
    elif relation.lower() == "includes":
        template = Template("Does $event1 temporally include $event2?")
    elif relation.lower() == "is included":
        template = Template("Is $event1 temporally included in $event2?")
    elif relation.lower() == "simultaneous":
        template = Template("Did $event1 happen simultaneously with $event2?")

    return template


def create_yn(event_pair, relation, relation_set):

    q_texts = []
    answers = []

    for r in relation_set:
        if r.lower() != "vague":
            template = get_temporal_question(r)

        q_texts.append(template.substitute(event1=event_pair[0], event2=event_pair[1]))

        if r.lower() == relation.lower():
            answers.append("Yes")
        else:
            answers.append("No")

    return q_texts, answers


def create_fr(event_pair, relation):

    # template = Template("What is the temporal relation between $event1 and $event2?")
    template = Template("When did $event1 happen in time compared to $event2?")
    q_text = template.substitute(event1=event_pair[0], event2=event_pair[1])

    answer = [relation]

    return q_text, answer


def create_chain(doc_pair_relations, trans_triples, inverse):
    doc_chains = []
    for i in range(len(doc_pair_relations)):
        pairs_chains = {}
        for pair in doc_pair_relations[i]:
            # print("Pair:", pair)
            # Check for existing tranitivity chain for pair (x,y)
            existing_transitivity = False
            for triple in trans_triples[i]:
                if pair == triple[2]:
                    # print("found")
                    # print(triple)
                    chain = {"num_facts": 2,
                             "reasoning_steps": 1,
                             "chain": [[triple[0],
                                        {"relation_type": doc_pair_relations[i][triple[0]], "relation_property": ""}],
                                       [triple[1],
                                        {"relation_type": doc_pair_relations[i][triple[1]], "relation_property": ""}]],
                             "goal_chain": [[triple[0][0], triple[0][1],
                                             {"relation_type": doc_pair_relations[i][triple[0]],
                                              "relation_property": ""}], [triple[1][0], triple[1][1], {
                                 "relation_type": doc_pair_relations[i][triple[1]], "relation_property": ""}]]
                             }
                    # print([triple[0], triple[1]])
                    existing_transitivity = True

                # Check for existing transitivity for (y,x)
                if (pair[1], pair[0]) == triple[2]:
                    # print("Pair:", pair, "triple:", triple)
                    chain = {"num_facts": 2,
                             "reasoning_steps": 2,
                             "chain": [[triple[0],
                                        {"relation_type": doc_pair_relations[i][triple[0]], "relation_property": ""}],
                                       [triple[1],
                                        {"relation_type": doc_pair_relations[i][triple[1]], "relation_property": ""}]],
                             "goal_chain": [[triple[2][0], triple[2][1],
                                             {"relation_type": inverse[doc_pair_relations[i][triple[2]]],
                                              "relation_property": ""}]]
                             }
                    existing_transitivity = True

                if existing_transitivity:
                    break
            #     if pair not in list(reasoning_chain.keys()):
            #         print(pair)

            # Create the transitivity
            if not existing_transitivity:
                candidate_symmetry = {}
                # Get all pairs that include x
                for p in doc_pair_relations[i]:
                    if p != pair:
                        if p[0] == pair[0]:
                            candidate_symmetry[p[1]] = p
                        elif p[1] == pair[0]:
                            candidate_symmetry[p[0]] = p
                # print("Candidates for", pair[0], ":", candidate_symmetry)

                # Get pairs that include y
                for p2 in doc_pair_relations[i]:
                    if p2 != pair:
                        # And the second event in the pair is connected with x
                        if p2[0] == pair[1] and p2[1] in list(candidate_symmetry.keys()):
                            if candidate_symmetry[p2[1]][0] == pair[0]:
                                chain = {"num_facts": 2,
                                         "reasoning_steps": 2,
                                         "chain": [[candidate_symmetry[p2[1]],
                                                    {"relation_type": doc_pair_relations[i][candidate_symmetry[p2[1]]],
                                                     "relation_property": ""}], [p2, {
                                             "relation_type": doc_pair_relations[i][p2], "relation_property": ""}]],
                                         "goal_chain": [[p2[1], p2[0],
                                                         {"relation_type": inverse[doc_pair_relations[i][p2]],
                                                          "relation_property": ""}]]
                                         }
                            else:
                                chain = {"num_facts": 2,
                                         "reasoning_steps": 3,
                                         "chain": [[candidate_symmetry[p2[1]],
                                                    {"relation_type": doc_pair_relations[i][candidate_symmetry[p2[1]]],
                                                     "relation_property": ""}], [p2, {
                                             "relation_type": doc_pair_relations[i][p2], "relation_property": ""}]],
                                         "goal_chain": [[candidate_symmetry[p2[1]][1], candidate_symmetry[p2[1]][0], {
                                             "relation_type": inverse[doc_pair_relations[i][candidate_symmetry[p2[1]]]],
                                             "relation_property": ""}], [p[2][1], p[2][0], {
                                             "relation_type": inverse[doc_pair_relations[i][p2]],
                                             "relation_property": ""}]]
                                         }
                                existing_transitivity = True
                        elif p2[1] == pair[1] and p2[0] in list(candidate_symmetry.keys()):
                            chain = {"num_facts": 2,
                                     "reasoning_steps": 2,
                                     "chain": [[candidate_symmetry[p2[0]],
                                                {"relation_type": doc_pair_relations[i][candidate_symmetry[p2[0]]],
                                                 "relation_property": ""}], [p2, {
                                         "relation_type": doc_pair_relations[i][p2], "relation_property": ""}]],
                                     "goal_chain": [[candidate_symmetry[p2[0]][1], candidate_symmetry[p2[0]][0], {
                                         "relation_type": inverse[doc_pair_relations[i][candidate_symmetry[p2[0]]]],
                                         "relation_property": ""}]]
                                     }
                            existing_transitivity = True
                    if existing_transitivity:
                        break

            # If transivity could not be found, just add one inverse step
            if not existing_transitivity:
                chain = {"num_facts": 1,
                         "reasoning_steps": 1,
                         "chain": [[[pair[1], pair[0]],
                                    {"relation_type": inverse[doc_pair_relations[i][pair]], "relation_property": ""}]],
                         "goal_chain": [[pair[0], pair[1],
                                         {"relation_type": doc_pair_relations[i][pair], "relation_property": ""}]]
                         }

            pairs_chains[pair] = chain
        doc_chains.append(pairs_chains)
    return doc_chains
    #     doc_pairs.append(len(doc_pair_relations[i]))
    #     doc_chained_pairs.append(len(chain))


def build_data(ids, story_triplets, questions, objects_info):
    data = []
    for i in range(len(ids)):
        entry = {
            'identifier': ids[i],
            'seed_id': i,
            'story': "",
            'story_triplets': story_triplets[i],
            'questions': questions[i],
            'objects_info': objects_info[i]
        }
        data.append(entry)
    return data


def save_json(file_path, dataset_name, data_list):
    output = {
        'name': dataset_name,
        'data': data_list
    }
    with open(os.path.join(file_path, dataset_name), 'w') as f:
        json.dump(output, f, indent=4)

