import numpy as np
import time
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


portid = 1
rew = []
action1 = []

class QLearningEGreedy:

	def __init__(self, prot, learn_rate = 0.3, discount = 0.8, epsilon = 0):
		logging.info("QLearnging e-greedy")

		self.q_table	= np.zeros((2, 2))
		#self.q_table	= np.random.rand(2, 2) - 0.5
		self.discount   = discount
		self.learn_rate = learn_rate
		self.reward	    = 0.

		self.epsilon    = epsilon

		self.t		    = 1

		seed = int(time.time())
		np.random.seed(seed)

		_ = self.decision(prot)

		return
		
	def decision(self, prot, keep = False, force_switch = False):
		self.state = prot

		epsilon = self.epsilon
		self.t = self.t + 1

		if force_switch == True:
			action = 0 if prot == 1 else 1
			action1.append(action)

		elif keep == False:
			if np.random.rand() < epsilon:
				action = np.random.randint(2)
				logging.info("Random choice = {}".format(action))
				action1.append(action)
			else:
				action = np.argmax(self.q_table[self.state, :])
				action1.append(action)

		else:
			action = prot
			action1.append(action)
		
		self.action = action
		self.state_new = action
		
		return action
	
	def update_qtable(self, reward, dt = 0):
		self.q_table[self.state, self.action] = (1. - self.learn_rate) * self.q_table[self.state, self.action] + \
						self.learn_rate * (reward + self.discount * np.max(self.q_table[self.state_new, :]))

		logging.info("Reward = {}".format(reward))
		logging.info("QTable = \n{}".format(self.q_table))
		
		return

	def reset_qtable(self):
		self.q_table = np.random.rand(2, 2) - 0.5

		return

	
	# }}}

class decision_final:
	def calc_reward(self, curr, prev): # {{{
		if curr > prev:
			reward = curr / prev - 1. if prev > 0. else 0.
		else:
			reward = - (prev / curr - 1.) if curr > 0. else 0.
		if reward > 1. or reward < -1:
			reward = 1 if reward > 1 else -1

		if self.minmax == 0:	# min
			return reward * -5.
		elif self.minmax == 1:	# max
			return reward * 5.

	def aggr(self, aggr, list): # {{{
		array = np.array(list)
		if len(array) == 0:
			met = None
		elif aggr == 0:
			met = array.sum()
		elif aggr == 2: 
			met = array.max()
		elif aggr == 3:
			met = array.min()
		elif aggr == 4:
			met = array.var()
		elif aggr == 5:
			met = array.shape[0]

		return met

	def __init__(self, metric, minmax):
		mode = 1
		global portid
		seed = int(time.time())
		np.random.seed(seed)
		portid = np.random.choice([0, 1], p = [0.5, 0.5])
		#f_init_prot = open("/tmp/init_prot.txt", "r")
		#init_prot = int(f_init_prot.readline().strip("\n"))
		#portid = init_prot

		##### MODE #####
		#f_mode = open("/tmp/prot.txt", "r")
		#mode = int(f_mode.readline().strip("\n"))
		#filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ram.log')
		logging.basicConfig(filename="out3.log", filemode = 'w', level = logging.INFO)
		self.metric = metric
		self.minmax = minmax
		#self.aggregation = aggr
	# Aggregator
	
		#self.backlog_file = backlog_file

		if mode == 0 or mode == 1:
			portid = mode
			mode = 2
		################

		reward = 0.
		prev_portid = portid

		# ML modules
		if mode == 2:
			somac = QLearningEGreedy(portid)

			is_transition = lambda _p, _pp: 1. if _p != _pp else 0.

		dt = -2 # delta time since last protocol switch
		t  = 0

		logging.info("Decision block as Coordinator")
		logging.info("Init prot: {}".format("CSMA" if portid == 0 else "TDMA"))
		prev = -1
		prev_prev = -1
		target_metric = {}

		while t < 40: # {{{
			logging.info("Active protocol: {}".format("CSMA" if portid == 0 else "TDMA"))

			if np.any(np.equal(metric, None)) == False: # {{{
				log_dict = {}
					#if t > 0:
						#log_dict = np.load(self.backlog_file).item()

				log_dict[t] = {	"prot": portid, "metrics": metric,
							"transition": is_transition(portid, prev_portid),
							"dt": dt }
				dt = dt + 1
				#dt = 2
				prev_portid = portid
					#np.save(self.backlog_file, log_dict)

				#target_metric[t] = log_dict[t]["metrics"]
				target_metric[t] = metric[t]
				logging.info("Target metric = {}".format(target_metric[t]))


					# Decision SOMAC {{{
					# Guarantees two decision are not done in a row
					# This is the mode code for SOMAC

				if (mode == 2 or mode == 3): 

					if dt > 0:
						if dt == 2:
							reward = self.calc_reward(target_metric[t], target_metric[t-2])
							logging.info("Reward = {}".format(reward))
							rew.append(reward)
						elif dt == 3:
							reward = self.calc_reward(target_metric[t], target_metric[t-3])
							logging.info("Reward = {}".format(reward))
							rew.append(reward)
						else:
							reward = self.calc_reward(target_metric[t], target_metric[t-1])
							if reward >= 0:
								reward = 0
								rew.append(reward)
							else:
								reward = reward
								rew.append(reward)
							logging.info("Reward = {}".format(reward))

						logging.info("dt = {}".format(dt))
						somac.update_qtable(reward, dt)

						decision = somac.decision(portid)
						logging.info("Decision: {}".format(decision))


						if portid != decision: 
							portid = decision
							dt = 0
						else:
							logging.info("No decision: protocol was switched last time")
					# }}}
					else:
						logging.info("Mode: {}".format("pure-CSMA" if mode == 0 else "pure-TDMA"))
					t = t + 1
				else:
					logging.info("Metrics contain None")
metric1=[]
for x in range(40):
	#if x >= 15:
	metric1.append(10*x) 
	#else:
		#metric1.append(10*x)  

q1 = decision_final(metric1, 1)

plt.figure(num=1)
plt.grid(True)
line1, = plt.plot(metric1/np.max(metric1), 'g', label = 'Normalised Metric')
plt.legend(handles=[line1], loc='lower left')
plt.xlabel('Time index')
plt.ylabel('Metric value(Normalised by max)')
#plt.plot(metric1)
#plt.show()	

#plt.plot(action1/np.max(action1))
plt.figure(num=2)
plt.grid(True)
line2, = plt.plot(action1, label = 'Action Value')	
#plt.annotate('CSMA',
            #xy=(30, 410), xycoords='figure pixels')
#plt.show()	
#plt.annotate('TDMA',
            #xy=(30, 235), xycoords='figure pixels')
#plt.plot(rew/np.max(rew))
#plt.figure(num=2)
line3, = plt.plot(rew, label = 'Reward value')
plt.xlabel('Time index')
plt.ylabel('Value')	
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})


plt.show()
