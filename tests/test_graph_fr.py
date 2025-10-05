import argparse
import os
import sys

import torch


from graph_fr import get_graph

from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.learners import TorchLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor


(
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
) = get_graph()

graph.detach()

class DummyLearner(TorchLearner):
    def __init__(self, *pre):
        TorchLearner.__init__(self, *pre)

    def forward(self, x):
        """
        Returns neutral scores for all 6 temporal relations.
        All labels start at -1000, letting the ILP solver decide based on constraints.
        
        Args:
            x: array of indices for questions

        Returns:
            tensor of shape (len(x), 6) with -1000 for all labels:
            [before, after, simultaneous, is_included, includes, vague]
        """
        result = torch.full((len(x), 6), -1000.0)
        return result
    
