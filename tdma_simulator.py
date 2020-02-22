#%%
# library imports
import numpy as np
from mac_simulator import * 

# initialisations
num_nodes = 5
num_packets = 5
frame_duration = 1
start_time = 0
sim_end_time = start_time + frame_duration
events = generate_events(num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time, round=3)
nodeID = np.zeros((1,events.shape[1]))
for i in range(events.shape[1]):
    nodeID[0][i] = np.random.randint(num_nodes)
events = np.append(events,nodeID,axis=0)
print('ran')
#%%
# fuctions
def tdma_simulator(simevents=None , frame_duration = 10 , num_nodes = 10 , packet_time = 0.01 , printFlag=0 , start_time = 0):
    """Function for implementing the event based simulator with TDMA Protocol.

    Keyword Arguments:
        simEvents {numpy array} -- First row - event times, second row - state IDs, third row - packet IDs
        num_nodes {int} -- Number of nodes contending currently in the network. (default: {10})
        packet_time {float} -- Duration of the packet transmission (default: {0.01}) . Unit - seconds
        printFlag {int} -- all the debug prints can be enabled with this flag (default: {0})
    Returns:
        [type] -- [description]
    """
    
    poll_period = packet_time * num_nodes
    no_of_rounds = frame_duration/poll_period
    poll_period_intervals = np.linspace(start_time,start_time+frame_duration,no_of_rounds+1)

    for round_iter in range(round_trip_intervals):
        curr_round_start = round_trip_intervals(round_iter)
        curr_round_end = curr_round_start + round_trip_period
        curr_round_events = []
        if(simevents[0,:])
        














    return round_trip_intervals




# test scripts

round_trip_intervals = tdma_simulator(frame_duration=1.0)

# %%
