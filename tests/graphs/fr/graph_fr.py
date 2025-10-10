from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, exactL, ifL, orL


def get_graph(
    symmetric: bool = False,
    transitive_determin: bool = False,
    transitive_non_determin: bool = False,
):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    transitive = None
    inverse = None
    tran_quest1 = None
    tran_quest2 = None
    tran_quest3 = None
    inv_question1 = None
    inv_question2 = None

    with Graph("temporal_QA_rule") as graph:
        story = Concept(name="story")
        question = Concept(name="question")
        (story_contain,) = story.contains(question)

        before = question(name="before")
        after = question(name="after")
        simultaneous = question(name="simultaneous")
        is_included = question(name="is included")
        includes = question(name="includes")
        vague = question(name="vague")

        ifL(
            question("q"),
            exactL(
                before(path=("q",)),
                after(path=("q",)),
                simultaneous(path=("q",)),
                is_included(path=("q",)),
                includes(path=("q",)),
                vague(path=("q",)),
            ),
        )

        # if inverse:
        #     # Inverse Constrains
        #     inverse = Concept(name="inverse")
        #     inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

        #     inverse_list1 = [(before, after), (includes, is_included)]
        #     for ans1, ans2 in inverse_list1:
        #         ifL(
        #             andL(inverse("s"), ans1(path=("s", inv_question1))),
        #             ans2(path=("s", inv_question2)),
        #         )

        #         ifL(
        #             andL(inverse("s"), ans2(path=("s", inv_question1))),
        #             ans1(path=("s", inv_question2)),
        #         )

        if symmetric:
            inverse = Concept(name="inverse")
            inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

            # symmetric - if q1 has label, q2 has same label
            inverse_list2 = [(simultaneous, simultaneous), (vague, vague)]
            for ans1, ans2 in inverse_list2:
                ifL(
                    andL(inverse("s"), ans1(path=("s", inv_question1))),
                    ans2(path=("s", inv_question2)),
                )

        if transitive_determin:
            transitive = Concept(name="transitive")
            tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

            # A rel B & B rel C -> A rel C
            transitive_1 = [before, after, includes, is_included, simultaneous, vague]
            for rel in transitive_1:
                ifL(
                    andL(
                        transitive("t"),
                        rel(path=("t", tran_quest1)),
                        rel(path=("t", tran_quest2)),
                    ),
                    rel(path=("t", tran_quest3)),
                )

        if transitive_non_determin:
            transitive = Concept(name="transitive")
            tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

            # A<B & B includes C => [A<C, A includes C, A vague C]
            ifL(
                andL(
                    transitive("t"),
                    before(path=("t", tran_quest1)),
                    includes(path=("t", tran_quest2)),
                ),
                orL(
                    before(path=("t", tran_quest3)),
                    includes(path=("t", tran_quest3)),
                    vague(path=("t", tran_quest3)),
                ),
            )

    return (
        graph,
        story,
        question,
        transitive,
        inverse,
        before,
        after,
        simultaneous,
        is_included,
        includes,
        vague,
        story_contain,
        tran_quest1,
        tran_quest2,
        tran_quest3,
        inv_question1,
        inv_question2,
    )
