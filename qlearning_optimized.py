import numpy as np
import time
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


#class QLearningBoltzmann:
#
#    def __init__(self, prot, learn_rate=0.3, discount=0.8, T=0.1):
#        self.q_table = np.zeros((2, 2))
#
#        self.prob_table = np.zeros((2, 2)) + 0.5
#        self.discount = discount
#        self.learn_rate = learn_rate
#        self.reward = 0.
#
#        self.T = T
#
#        seed = int(time.time())
#        np.random.seed(seed)
#
#        _ = self.decision(prot, keep=True)
#
#        logging.info("QLearnging Boltzmann")
#        logging.info("T = {}, learn_rate = {}, discount factor = {}".format(
#            self.T, self.learn_rate, self.discount))
#
#        return
#
#    def decision(self, prot, keep=False, force_switch=False):
#        self.state = prot
#
#        if force_switch == True:
#            action = 0 if prot == 1 else 1
#        elif keep == False:
#            action = np.random.choice(
#                np.array([0, 1]),
#                p=self.prob_table[self.state]
#            )
#            # prob.append(self.prob_table[self.state])
#        else:
#            action = prot
#
#        logging.info("Choice = {}".format(action))
#
#        self.action = action
#        self.state_new = action
#
#        return action
#
#    def update_qtable(self, reward, dt):
#        self.q_table[self.state, self.action] = (1. - self.learn_rate) * self.q_table[self.state, self.action] + \
#            self.learn_rate * (reward + self.discount *
#                               np.max(self.q_table[self.state_new, :]))
#
#        T = self.T
#        num = np.exp(self.q_table / T)
#
#        sum_cols = np.sum(num, 1)
#        for row in range(num.shape[0]):
#            num[row, :] = num[row, :] / sum_cols[row]
#
#        self.prob_table = num
#
#        logging.info("Temperature = {}".format(T))
#        logging.info("Reward = {}".format(reward))
#        logging.info("QTable = \n{}".format(self.q_table))
#        logging.info("Prob Table = \n{}".format(self.prob_table))
#
#        return
#
#    def reset_qtable(self):
#        self.q_table = np.zeros((2, 2))
#        self.prob_table = np.zeros((2, 2)) + 0.5
#
#        return


class QLearningEGreedy:

    def __init__(self, prot, learn_rate=0.3, discount=0.8, epsilon=0):

        self.q_table = np.zeros((2, 2))
        #self.q_table	= np.random.rand(2, 2) - 0.5
        self.discount = discount
        self.learn_rate = learn_rate
        self.reward = 0.

        self.epsilon = epsilon

        self.t = 1

        seed = int(time.time())
        np.random.seed(seed)

        _ = self.decision(prot)

        return

    def decision(self, prot, keep=False, force_switch=False):
        self.state = prot

        epsilon = self.epsilon
        self.t = self.t + 1

        if force_switch == True:
            action = 0 if prot == 1 else 1
            # action1.append(action)

        elif keep == False:
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
                # action1.append(action)
            else:
                action = np.argmax(self.q_table[self.state, :])
                # action1.append(action)
        else:
            action = prot
            # action1.append(action)

        self.action = action
        self.state_new = action

        return action

    def update_qtable(self, reward, dt=0):
        self.q_table[self.state, self.action] = (1. - self.learn_rate) * self.q_table[self.state, self.action] + \
            self.learn_rate * (reward + self.discount *
                               np.max(self.q_table[self.state_new, :]))
        logging.info("Reward = {}".format(reward))
        logging.info("QTable = \n{}".format(self.q_table))
        return

    def reset_qtable(self):
        self.q_table = np.random.rand(2, 2) - 0.5
        return


class decision_final:
    def calc_reward(self, curr, prev):  # {{{
        if curr > prev:
            reward = curr / prev - 1. if prev > 0. else 0.
        else:
            reward = - (prev / curr - 1.) if curr > 0. else 0.
        if reward > 1. or reward < -1:
            reward = 1 if reward > 1 else -1

        if self.minmax == 0:  # min
            return reward * -5.
        elif self.minmax == 1:  # max
            return reward * 5.

    def __init__(self, metric, minmax, mode=0, g_dt=-2):
        seed = int(time.time())
        np.random.seed(seed)
        protid = np.random.choice([0, 1], p = [0.5, 0.5])
#        logging.basicConfig(filename="out3.log",
#                            filemode='w', level=logging.INFO)
        self.metric = metric
        self.minmax = minmax

        if mode == 0 or mode == 1:
            protid = mode
            mode = 2

        reward = 0.

        # ML modules
        if mode == 2:
            somac = QLearningEGreedy(protid)
#        if mode == 3:
#            somac = QlearningBoltzmann(protid)
        decision = protid
        
        if np.any(np.equal(metric, None)) == False:
            g_dt = g_dt + 1
            if ((mode == 2) and (len(metric) >= 4)):
                if g_dt > 0:
                    if g_dt == 2:
                        reward = self.calc_reward(metric[-1], metric[-3])
                    elif g_dt == 3:
                        reward = self.calc_reward(metric[-1], metric[-4])
                    else:
                        reward = self.calc_reward(metric[-1], metric[-2])
    #						if reward >= 0:
    #                            reward = 0
    #                        else:
    #                            reward = reward
    
                    somac.update_qtable(reward, g_dt)
                    decision = somac.decision(protid)
    
                    if protid != decision:
                        protid = decision
                        g_dt = 0
                else:
                    logging.info(
                        "No decision: protocol was switched last time")
            else:
                logging.info("Mode: {}".format(
                    "pure-CSMA" if mode == 0 else "pure-TDMA"))
        else:
            logging.info("Metrics contain None")
        self.decision = protid
        self.g_dt = g_dt

        return

# metric1=[]
#protocol = decision_final(metric1, 1)
