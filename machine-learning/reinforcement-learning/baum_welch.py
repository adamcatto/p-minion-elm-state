"""
author: Adam Catto

description: an implementation of the Baum-Welch algorithm for HMM
parameter estimation
"""

import numpy as np
import pandas as pd
from pomdpy.pomdpy.pomdp import model


class HiddenMarkovModel:
    def __init__(self, states, observations, initial_belief):
        self.states = states
        self.observations = observations
        self.initial_belief = np.array([1 / len(states)] * len(states))
        self.transition_probabilities = np.zeros((len(states), len(states)))
        self.information_matrix = np.zeros((len(states), len(observations)))
        self.observation_sequence = np.array([])

    
    def compute_conditional_obs_probs(self, data: pd.DataFrame):
        belief, tp, info_matrix = self.initial_belief, self.transition_probabilities, self.information_matrix
        for i, row in data.iterrows():
            pass
        return []


    def estimate_model_parameters(self, data: pd.DataFrame, tolerance: float = 0.8, max_iters: int = 500):
        initial_belief = self.initial_belief
        observation_probabilities = self.compute_conditional_obs_probs(data)
        
