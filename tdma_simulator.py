#%%
# library imports
import numpy as np

# initialisations


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
    
    round_trip_period = packet_time * num_nodes
    no_of_rounds = frame_duration/round_trip_period
    round_trip_intervals = np.linspace(start_time,start_time+frame_duration,no_of_rounds+1)
    return round_trip_intervals




# test scripts

round_trip_intervals = tdma_simulator(frame_duration=1.0)

# %%
