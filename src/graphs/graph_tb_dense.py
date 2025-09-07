from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, exactL, existsL, ifL

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()

with Graph("temporal_QA_rule") as graph:
    # Group of sentence
    story = Concept(name="story")
    question = Concept(name="question")
    (story_contain,) = story.contains(question)

    # labels
    before = question(name="before")
    after = question(name="after")
    # equal
    simultaneous = question(name="simultaneous")
    # x is included in y = x during y
    is_included = question(name="is included")
    # converse of x during y
    includes = question(name="includes")
    vague = question(name="vague")

    output_for_loss = question(name="output_for_loss")

    # opposite concepts
    exactL(before, after)
    exactL(includes, is_included)

    # Inverse Constrains
    inverse = Concept(name="inverse")
    inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

    # First inverse relation, allow inverse back and forth
    inverse_list1 = [(before, after), (includes, is_included)]

    for ans1, ans2 in inverse_list1:
        ifL(andL(ans1("x"), existsL(inverse("s", path=("x", inverse)))), ans2(path=("s", inv_question2)))

        ifL(andL(ans2("x"), existsL(inverse("s", path=("x", inverse)))), ans1(path=("s", inv_question2)))

    # symmetric
    inverse_list2 = [(simultaneous, simultaneous), (vague, vague)]
    for ans1, ans2 in inverse_list2:
        ifL(andL(ans1("x"), existsL(inverse("s", path=("x", inverse)))), ans2(path=("s", inv_question2)))

    # Transitive constrains
    transitive = Concept(name="transitive")
    tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

    # if A & B have relation x, B & C have relation x, then A & C have relation x
    transitive_1 = [before, after, includes, is_included]
    for rel in transitive_1:
        ifL(
            andL(rel("x"), existsL(transitive("t", path=("x", transitive))), rel(path=("t", tran_quest2))),
            rel(path=("t", tran_quest3)),
        )

    # A<B & B includes C => A<C
    ifL(
        andL(before("x"), existsL(transitive("t", path=("x", transitive))), includes(path=("t", tran_quest2))),
        before(path=("t", tran_quest3)),
    )

    # A>B & B includes C => A>C
    ifL(
        andL(after("x"), existsL(transitive("t", path=("x", transitive))), includes(path=("t", tran_quest2))),
        after(path=("t", tran_quest3)),
    )

    # A is_included B & B<C => A<C
    ifL(
        andL(is_included("x"), existsL(transitive("t", path=("x", transitive))), before(path=("t", tran_quest2))),
        before(path=("t", tran_quest3)),
    )

    # A is_included B & B>C => A>C
    ifL(
        andL(is_included("x"), existsL(transitive("t", path=("x", transitive))), after(path=("t", tran_quest2))),
        after(path=("t", tran_quest3)),
    )

if __name__ == "__main__":
    graph.visualize("graph-temporal")
