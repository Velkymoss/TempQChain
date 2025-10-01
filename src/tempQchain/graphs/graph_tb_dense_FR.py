from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, exactL, existsL, ifL, orL

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()

# with Graph("temporal_QA_rule") as graph:
#     # Group of sentence
#     story = Concept(name="story")
#     question = Concept(name="question")
#     (story_contain,) = story.contains(question)

#     # labels
#     before = question(name="before")
#     after = question(name="after")
#     # equal
#     simultaneous = question(name="simultaneous")
#     # x is included in y = x during y
#     is_included = question(name="is included")
#     # converse of x during y
#     includes = question(name="includes")
#     vague = question(name="vague")

#     output_for_loss = question(name="output_for_loss")

#     # opposite concepts
#     exactL(before, after, name="before_after-inverse")
#     exactL(includes, is_included)

#     # Inverse Constrains
#     inverse = Concept(name="inverse")
#     inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

#     # First inverse relation, allow inverse back and forth
#     inverse_list1 = [(before, after), (includes, is_included)]

#     for ans1, ans2 in inverse_list1:
#         ifL(andL(ans1("x"), existsL(inverse("s", path=("x", inverse)))), ans2(path=("s", inv_question2)))

#         ifL(andL(ans2("x"), existsL(inverse("s", path=("x", inverse)))), ans1(path=("s", inv_question2)))

#     # symmetric
#     inverse_list2 = [(simultaneous, simultaneous), (vague, vague)]
#     for ans1, ans2 in inverse_list2:
#         ifL(andL(ans1("x"), existsL(inverse("s", path=("x", inverse)))), ans2(path=("s", inv_question2)))

#     ########################################################################################
#     ############################ Transitive ##################################################
#     transitive = Concept(name="transitive")
#     tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

#     # A rel B & B rel C -> A rel C
#     transitive_1 = [before, after, includes, is_included, simultaneous, vague]
#     for rel in transitive_1:
#         ifL(
#             andL(rel("x"), existsL(transitive("t", path=("x", transitive))), rel(path=("t", tran_quest2))),
#             rel(path=("t", tran_quest3)),
#         )

#     # #######################rule 2 #####################################################
#     # A rel B & B=C -> A rel C
#     for rel in [before, after, includes, is_included]:
#         ifL(
#             andL(rel("x"), existsL(transitive("t", path=("x", transitive))), simultaneous(path=("t", tran_quest2))),
#             rel(path=("t", tran_quest3)),
#         )

#     ############################ before ##################################################
#     if CONSTRAIN_ACTIVE:
#         # A<B & B > C => A any C
#         ifL(
#             andL(before("x"), existsL(transitive("t", path=("x", transitive))), after(path=("t", tran_quest2))),
#             orL(
#                 before(path=("t", tran_quest3)),
#                 after(path=("t", tran_quest3)),
#                 includes(path=("t", tran_quest3)),
#                 is_included(path=("t", tran_quest3)),
#                 simultaneous(path=("t", tran_quest3)),
#                 vague(path=("t", tran_quest3)),
#             ),
#         )

#     # A<B & B includes C => [A<C, A includes C, A vague C]
#     ifL(
#         andL(before("x"), existsL(transitive("t", path=("x", transitive))), includes(path=("t", tran_quest2))),
#         orL(
#             before(path=("t", tran_quest3)),
#             includes(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     # A<B & B is_included C -> [before, is_included, vague]
#     ifL(
#         andL(before("x"), existsL(transitive("t", path=("x", transitive))), is_included(path=("t", tran_quest2))),
#         orL(
#             before(path=("t", tran_quest3)),
#             is_included(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     ########################### after ####################################################
#     # A>B & B includes C => [after, includes, vague]
#     ifL(
#         andL(after("x"), existsL(transitive("t", path=("x", transitive))), includes(path=("t", tran_quest2))),
#         orL(
#             after(path=("t", tran_quest3)),
#             includes(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     if CONSTRAIN_ACTIVE:
#         # A>B & B<C -> A any C
#         ifL(
#             andL(after("x"), existsL(transitive("t", path=("x", transitive))), before(path=("t", tran_quest2))),
#             orL(
#                 before(path=("t", tran_quest3)),
#                 after(path=("t", tran_quest3)),
#                 includes(path=("t", tran_quest3)),
#                 is_included(path=("t", tran_quest3)),
#                 simultaneous(path=("t", tran_quest3)),
#                 vague(path=("t", tran_quest3)),
#             ),
#         )

#     # A>B & B is_included C -> [after, is_included, vague]
#     ifL(
#         andL(after("x"), existsL(transitive("t", path=("x", transitive))), before(path=("t", tran_quest2))),
#         orL(
#             after(path=("t", tran_quest3)),
#             is_included(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     ######################## is included ##################################################
#     # A is_included B & B<C => [before, is_included, vague]
#     ifL(
#         andL(is_included("x"), existsL(transitive("t", path=("x", transitive))), before(path=("t", tran_quest2))),
#         orL(
#             before(path=("t", tran_quest3)),
#             is_included(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     # A is_included B & B>C => [after, is_included, vague]
#     ifL(
#         andL(is_included("x"), existsL(transitive("t", path=("x", transitive))), after(path=("t", tran_quest2))),
#         orL(
#             before(path=("t", tran_quest3)),
#             is_included(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     if CONSTRAIN_ACTIVE:
#         # A is_included B & B includes C => A any C
#         ifL(
#             andL(is_included("x"), existsL(transitive("t", path=("x", transitive))), includes(path=("t", tran_quest2))),
#             orL(
#                 before(path=("t", tran_quest3)),
#                 after(path=("t", tran_quest3)),
#                 includes(path=("t", tran_quest3)),
#                 is_included(path=("t", tran_quest3)),
#                 simultaneous(path=("t", tran_quest3)),
#                 vague(path=("t", tran_quest3)),
#             ),
#         )

#     ############################ includes ###########################################################
#     # A includes B & B < C -> [before, includes, vague]
#     ifL(
#         andL(includes("x"), existsL(transitive("t", path=("x", transitive))), before(path=("t", tran_quest2))),
#         orL(
#             before(path=("t", tran_quest3)),
#             includes(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     # A includes B & B > C -> [after, includes, vague]
#     ifL(
#         andL(includes("x"), existsL(transitive("t", path=("x", transitive))), after(path=("t", tran_quest2))),
#         orL(
#             after(path=("t", tran_quest3)),
#             includes(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     # A includes B & B is_included C -> A any C
#     ifL(
#         andL(includes("x"), existsL(transitive("t", path=("x", transitive))), is_included(path=("t", tran_quest2))),
#         orL(
#             includes(path=("t", tran_quest3)),
#             is_included(path=("t", tran_quest3)),
#             simultaneous(path=("t", tran_quest3)),
#             vague(path=("t", tran_quest3)),
#         ),
#     )

#     #######################simultaneous#####################################
#     # A = B & B rel C -> A rel C
#     for rel in [before, after, includes, is_included]:
#         ifL(
#             andL(simultaneous("x"), existsL(transitive("t", path=("x", transitive))), rel(path=("t", tran_quest2))),
#             rel(path=("t", tran_quest3)),
#         )



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


    inverse_list1 = [(before, after), (includes, is_included)]

    for ans1, ans2 in inverse_list1:
        ifL(
            andL(inverse("s"), ans1(path=("s", inv_question1))),
            ans2(path=("s", inv_question2)),
        )

        ifL(
            andL(inverse("s"), ans2(path=("s", inv_question1))),
            ans1(path=("s", inv_question2)),
        )

    # symmetric - if q1 has label, q2 has same label
    inverse_list2 = [(simultaneous, simultaneous), (vague, vague)]
    for ans1, ans2 in inverse_list2:
        ifL(
            andL(inverse("s"), ans1(path=("s", inv_question1))),
            ans2(path=("s", inv_question2)),
        )

    ########################################################################################
    ############################ Transitive ##################################################
    transitive = Concept(name="transitive")
    tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(
        arg11=question, arg22=question, arg33=question
    )

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

    #######################rule 2 #####################################################
    # A rel B & B=C -> A rel C
    for rel in [before, after, includes, is_included]:
        ifL(
            andL(
                transitive("t"),
                rel(path=("t", tran_quest1)),
                simultaneous(path=("t", tran_quest2)),
            ),
            rel(path=("t", tran_quest3)),
        )

    ############################ before ##################################################
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

    # A<B & B is_included C -> [before, is_included, vague]
    ifL(
        andL(
            transitive("t"),
            before(path=("t", tran_quest1)),
            is_included(path=("t", tran_quest2)),
        ),
        orL(
            before(path=("t", tran_quest3)),
            is_included(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    ########################### after ####################################################
    # A>B & B includes C => [after, includes, vague]
    ifL(
        andL(
            transitive("t"),
            after(path=("t", tran_quest1)),
            includes(path=("t", tran_quest2)),
        ),
        orL(
            after(path=("t", tran_quest3)),
            includes(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    # A>B & B is_included C -> [after, is_included, vague]
    ifL(
        andL(
            transitive("t"),
            after(path=("t", tran_quest1)),
            is_included(path=("t", tran_quest2)),
        ),
        orL(
            after(path=("t", tran_quest3)),
            is_included(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    ######################## is included ##################################################
    # A is_included B & B<C => [before, is_included, vague]
    ifL(
        andL(
            transitive("t"),
            is_included(path=("t", tran_quest1)),
            before(path=("t", tran_quest2)),
        ),
        orL(
            before(path=("t", tran_quest3)),
            is_included(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    # A is_included B & B>C => [after, is_included, vague]
    ifL(
        andL(
            transitive("t"),
            is_included(path=("t", tran_quest1)),
            after(path=("t", tran_quest2)),
        ),
        orL(
            after(path=("t", tran_quest3)),
            is_included(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    ############################ includes ###########################################################
    # A includes B & B < C -> [before, includes, vague]
    ifL(
        andL(
            transitive("t"),
            includes(path=("t", tran_quest1)),
            before(path=("t", tran_quest2)),
        ),
        orL(
            before(path=("t", tran_quest3)),
            includes(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    # A includes B & B > C -> [after, includes, vague]
    ifL(
        andL(
            transitive("t"),
            includes(path=("t", tran_quest1)),
            after(path=("t", tran_quest2)),
        ),
        orL(
            after(path=("t", tran_quest3)),
            includes(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    # A includes B & B is_included C -> [includes, is_included, simultaneous, vague]
    ifL(
        andL(
            transitive("t"),
            includes(path=("t", tran_quest1)),
            is_included(path=("t", tran_quest2)),
        ),
        orL(
            includes(path=("t", tran_quest3)),
            is_included(path=("t", tran_quest3)),
            simultaneous(path=("t", tran_quest3)),
            vague(path=("t", tran_quest3)),
        ),
    )

    #######################simultaneous#####################################
    # A = B & B rel C -> A rel C
    for rel in [before, after, includes, is_included]:
        ifL(
            andL(
                transitive("t"),
                simultaneous(path=("t", tran_quest1)),
                rel(path=("t", tran_quest2)),
            ),
            rel(path=("t", tran_quest3)),
        )
