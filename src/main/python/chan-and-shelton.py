"""
An implementation of the basic model described in Chan, Nicholas Tung, and Christian Shelton.
"An electronic market-maker." (2001).
"""

import numpy as np
import pandas as pd

"""
Simulation events
"""
EVENT_PRICE_CHANGE_UP    = 0
EVENT_PRICE_CHANGE_DOWN  = 1
EVENT_UNINFORMED_BUY     = 2
EVENT_UNINFORMED_SELL    = 3
EVENT_INFORMED_ARRIVAL   = 4

EVENT_ALL = \
    [EVENT_PRICE_CHANGE_DOWN, EVENT_PRICE_CHANGE_UP, EVENT_UNINFORMED_SELL,
         EVENT_UNINFORMED_BUY, EVENT_INFORMED_ARRIVAL]

""" 
Default simulation parameters.
"""

INITIAL_PRICE = 200
MAX_T = 150

PROB_PRICE = 0.2

PROB_PRICE_UP = PROB_PRICE
PROB_PRICE_DOWN = PROB_PRICE

PROB_UNINFORMED = 0.1

PROB_UNINFORMED_BUY = PROB_UNINFORMED
PROB_UNINFORMED_SELL = PROB_UNINFORMED

PROB_INFORMED = 0.4

ALL_PROB = [PROB_PRICE_DOWN, PROB_PRICE_UP, PROB_UNINFORMED_BUY, PROB_UNINFORMED_SELL, PROB_INFORMED]


class MarketMakerPolicy:
    """
        A class representing a policy for a market-making strategy.
        The policy specifies a change in price given an order imbalance.
    """

    def __init__(self):
        pass

    def price_delta(self, imbalance):
        pass

    def update(self, s, a, r_, s_, a_):
        pass


class ThresholdPolicy(MarketMakerPolicy):
    """
    A class representing a market-policy in which we decrease (increase) the price
    when it reaches the positive (negative) of a pre-specified integer threshold.
    """

    def __init__(self, threshold):
        """
        :param threshold: A positive integer representing the maximum absolute order-imbalance.
        """
        self.threshold = threshold

    def price_delta(self, imbalance):
        """
        :param imbalance:  The currently observed order-imbalance
        :return: The integer change in the market-maker's quote
        """
        if imbalance == +self.threshold:
            return -1
        elif imbalance == -self.threshold:
            return +1
        else:
            return 0


class MarketSimulation:
    """
    A class representing an intraday simulation of a simple high-frequency order-driven market.
    """

    def __init__(self, mm_policy, max_t=MAX_T, initial_price=INITIAL_PRICE, probabilities=ALL_PROB):
        """
        :param mm_policy:           The market-making policy
        :param max_t:               The duration of the trading day in integer units
        :param initial_price:       The initial price of the asset
        :param probabilities:       The event probabilities for each element of EVENT_ALL
        """
        self.max_t = max_t
        self.probabilities = probabilities
        self.initial_price = initial_price
        self.mm_policy = mm_policy

    def simulate_events(self):
        """ Return a randomly-generated sequence of events covering the entire trading day.
        The events are chosen from EVENT_ALL"""
        return np.random.choice(EVENT_ALL, p=self.probabilities, size=self.max_t)

    def simulate_fundamental_price(self, events):
        """ Given a sequence events, return a vector representing the time-series of
        the fundamental price. """
        price_changes = np.zeros(self.max_t)
        price_changes[events == EVENT_PRICE_CHANGE_DOWN] = -1
        price_changes[events == EVENT_PRICE_CHANGE_UP] = +1
        return self.initial_price + np.cumsum(price_changes)

    def simulate_uninformed_orders(self, events):
        """ Given a sequence of events, return a vector representing the time-series of
        order-flow from uninformed traders. """
        orders = np.zeros(self.max_t)
        orders[events == EVENT_UNINFORMED_BUY] = +1
        orders[events == EVENT_UNINFORMED_SELL] = -1
        return orders

    def informed_strategy(self, current_price, mm_price):
        """
        Simulate an informed-trader
        :param current_price:   The current fundamental price
        :param mm_price:        The current market-maker quote
        :return:                The total shares demanded by the informed trader
        """
        if current_price < mm_price:
            return 1
        elif current_price > mm_price:
            return -1
        else:
            return 0

    def mm_reward(self, current_fundamental_price, mm_current_price, order_sign):
        """
        Compute the change in the profit accrued to the  market-maker
        :param current_fundamental_price:  The current fundamental price.
        :param mm_current_price:           The market-maker's current quote.
        :param order_sign:                 +1 if the most recent transaction was buy,
                                           -1 if it is a sell, 0 for no transaction.
        :return:
        """
        if order_sign > 0:
            return current_fundamental_price - mm_current_price
        elif order_sign < 0:
            return mm_current_price - current_fundamental_price
        else:
            return 0

    def simulate_market(self, events, uninformed_orders, fundamental_price):
        """
        Simulate the entire market over a single trading day.
        :param events:              A sequence of events chosen from ALL_PROB
        :param uninformed_orders:   The uninformed order-flow
        :param fundamental_price:   The time-series of the fundamental-price
        :return:                    A tuple containing:
                                        time-series of fundamental prices,
                                        time-series of market-maker's prices,
                                        time-series of order-imbalance, time-series of rewards,
                                        time-series of actions (changes to the market maker's quote).
        """

        mm_prices = np.zeros(self.max_t)
        order_imbalances = np.zeros(self.max_t)
        informed_orders = np.zeros(self.max_t)
        rewards = np.zeros(self.max_t);
        actions = np.zeros(self.max_t)

        t_mm = 0
        mm_current_price = self.initial_price

        for t in range(self.max_t):

            if events[t] == EVENT_INFORMED_ARRIVAL:
                order = self.informed_strategy(fundamental_price[t], mm_current_price)
                informed_orders[t] = order
            else:
                order = uninformed_orders[t]

            imbalance = np.sum(informed_orders[t_mm:t] + uninformed_orders[t_mm:t])

            mm_price_delta = self.mm_policy.price_delta(imbalance)
            if mm_price_delta != 0:
                t_mm = t
                mm_current_price += mm_price_delta

            order_imbalances[t] = imbalance
            mm_prices[t] = mm_current_price
            actions[t] = mm_price_delta
            rewards[t] = self.mm_reward(fundamental_price[t], mm_current_price, order)

            if t > 0:
                self.mm_policy.update(order_imbalances[t-1], actions[t-1], rewards[t],
                                        imbalance, mm_price_delta)

        return fundamental_price, mm_prices, order_imbalances, rewards, actions

    def run(self):
        """
        Run the simulation.
         :return:                    A tuple containing:
                                        time-series of fundamental prices,
                                        time-series of market-maker's prices,
                                        time-series of order-imbalance, time-series of rewards,
                                        time-series of actions (changes to the market maker's quote).       :return:
        """

        events = self.simulate_events()
        fundamental_price = self.simulate_fundamental_price(events)
        uninformed_orders = self.simulate_uninformed_orders(events)

        return self.simulate_market(events, uninformed_orders, fundamental_price)


class QTable:

    def __init__(self, table=None, all_actions=[-1, 0, +1], all_states=range(-2, 3)):
        self.actions = all_actions
        self.states = all_states
        self.num_states = len(all_states)
        self.num_actions = len(all_actions)
        self.state_offset = self.num_states / 2
        if table is None:
            self.Q = np.zeros((len(all_states), len(all_actions)))
        else:
            self.Q = table

    def state(self, imbalance):
        s = int(imbalance) + self.offset
        if s >= self.num_states:
            return self.num_states
        elif s < 0:
            return 0
        else:
            return s

    def action(self, price_delta):
        return int(price_delta) + 1

    def q_values(self, imbalance):
        return self.Q[self.state(imbalance), :]

    def q_value(self, imbalance, price_delta):
        return self.Q[self.state(imbalance), self.action(price_delta)]

    def as_DataFrame(self):
        return pd.DataFrame(self.Q,
                             columns=["$\Delta p=%s$" % a for a in self.actions],
                             index=self.states)


class SarsaLearner(QTable):

    def __init__(self, table=None, all_actions=[-1, 0, +1], all_states=range(-2, 3), alpha=0.01, gamma=0.0):
        QTable.__init__(table, all_actions, all_states)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, a, r_, s_, a_):
        self.Q[self.state(s), self.action(a)] += \
            self.alpha * (r_ + self.gamma * (self.q_value(s_, a_) - self.q_value(s, a)))


class LearningMarketMaker(MarketMakerPolicy, SarsaLearner):

    def __init__(self, all_actions=[01, 0, +1], all_states=range(-2, 3), alpha=0.01, gamma=0.99, epsilon=0.02):
        SarsaLearner.__init__(None, all_actions, all_states, alpha, gamma)
        MarketMakerPolicy.__init__(self)
        self.epsilon = epsilon

    def price_delta(self, s):
        if np.random.random() <= self.epsilon:
            action = np.random.choice([-1, 0, +1])
        else:
            values = self.q_values(s)
            max_value = np.max(values)
            action = np.random.choice(np.where(values == max_value)[0]) - 1
        return action


def expected_reward_by_state_action(policy, all_states, all_actions=[-1, 0, 1],
                                        probabilities=ALL_PROB, samples=1000):

    result = np.zeros((samples, len(all_states), len(all_actions)))

    for i in range(samples):

        simulation = MarketSimulation(policy, probabilities=probabilities)
        _, states, rewards, actions = simulation.run()

        result[i, :, :] = \
            np.reshape(
                [np.nanmean(rewards[(states == state) &
                                    (actions == action)]) \
                    for state in all_states for action in all_actions],
                (len(all_states), len(all_actions)))

    return QTable(np.nanmean(result, axis=0), all_actions, all_states)


