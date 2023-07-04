import copy

import numpy as np
from farmgym.v2.policy_api import Policy_API, Policy_helper
from farmgym_games.game_builder.run_farm import run_policy_xp
from farmgym_games.game_catalogue.farm1.farm import env as Farm1


class UCB:
    def __init__(self, actions):
        self.actions = actions
        self.scores = np.zeros(len(actions))
        self.counts = np.zeros(len(actions))
        self.total_counts = 0

    def select_action(self):
        ucb_scores = self.calculate_ucb()
        return np.argmax(ucb_scores)

    def calculate_ucb(self):
        ucb_scores = self.scores / (self.counts + 1e-8) + np.sqrt(2 * np.log(self.total_counts + 1) / (self.counts + 1e-8))
        return ucb_scores

    def update(self):
        policy_idx = self.select_action()
        policy = self.actions[policy_idx]
        reward, cost = run_policy_xp(farm, copy.deepcopy(policy))
        self.scores[policy_idx] += reward
        self.counts[policy_idx] += 1
        self.total_counts += 1

    def get_scores(self):
        scores = self.scores / (self.counts + 1e-8)
        ucb_scores = self.calculate_ucb()
        print(f"Current scores = {[np.round(s,2) for s in scores]}")
        print(f"Current counts = {self.counts}")
        print(f"Current ucb scores = {[np.round(s,2) for s in ucb_scores]}")


def create_expert(water_amount=1, delayed_policy=0):
    """
    Create an expert policy.
    In order to create different versions of the expert, we either
    change the defaut water amount, or delay all actions by a number of days
    """

    # Define delay to create delayed policies:
    d = 0
    # Define single policies
    water_soil_day1_5l = helper.create_water_soil(amount=water_amount, delay=0 + d, day=1)
    herbicide_day2 = helper.create_scatter_cide(amount=1, delay=0 + d, day=2)
    fertilize_day4 = helper.create_scatter_fert(amount=1, delay=0 + d, day=4)
    water_soil_day6_1l = helper.create_water_soil(amount=1, delay=0 + d, day=6)
    harvest_fruit_delay4 = helper.create_harvest_fruit(delay=4 + d)
    harvest_ripe_delay4 = helper.create_harvest_ripe(delay=4 + d)
    # Create expert policy by combining single ones
    policies = [
        water_soil_day1_5l,
        herbicide_day2,
        fertilize_day4,
        water_soil_day6_1l,
        harvest_fruit_delay4,
        harvest_ripe_delay4,
    ]
    combined_policy = Policy_API.combine_policies([policy.api for policy in policies])
    return combined_policy


farm = Farm1()
helper = Policy_helper(farm)

policy1 = create_expert(1, 0)
policy2 = create_expert(3, 0)
policy3 = create_expert(5, 0)
policy4 = create_expert(1, 5)
policy5 = create_expert(3, 5)
policy6 = create_expert(5, 5)

policies = [policy1, policy2, policy3, policy4, policy5, policy5]

bandit = UCB(policies)
for i in range(1000):
    if i % 50 == 0:
        print(f"Step = {i}")
        bandit.get_scores()
        print("____")
    bandit.update()
