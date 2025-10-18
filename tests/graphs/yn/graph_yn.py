from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, ifL


def get_graph(transitive_constraint: bool = False, symmetric_constraint: bool = False):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("spatial_QA_rule") as graph:
        # Group of sentence
        story = Concept(name="story")
        question = Concept(name="question")
        (story_contain,) = story.contains(question)

        answer_class = question(name="answer_class", ConceptClass=EnumConcept, values=["no", "yes"])

        symmetric = None
        s_quest1 = None
        s_quest2 = None
        transitive = None
        t_quest1 = None
        t_quest2 = None
        t_quest3 = None

        if symmetric_constraint:
            # Symmetric constraint
            symmetric = Concept(name="symmetric")
            s_quest1, s_quest2 = symmetric.has_a(arg1=question, arg2=question)

            ifL(
                andL(symmetric("s"), answer_class.yes(path=("s", s_quest1))),
                answer_class.yes(path=("s", s_quest2)),
            )

        if transitive_constraint:
            transitive = Concept(name="transitive")
            t_quest1, t_quest2, t_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

            ifL(
                andL(
                    transitive("t"),
                    answer_class.yes(path=("t", t_quest1)),
                    answer_class.yes(path=("t", t_quest2)),
                ),
                answer_class.yes(path=("t", t_quest3)),
            )

    return (
        graph,
        story,
        question,
        answer_class,
        symmetric,
        s_quest1,
        s_quest2,
        transitive,
        t_quest1,
        t_quest2,
        t_quest3,
        story_contain,
    )
