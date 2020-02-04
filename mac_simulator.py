# %%
# Add imports
import numpy as np
# %%


def simulator(num_nodes=10, num_packets=3, num_slots=10, sim_end_time=10):
    """Function for implementing the event based simulator.

    Keyword Arguments:
        num_nodes {int} -- Number of nodes in the simulation. (default: {10})
        num_packets {int} -- Number of packets at each node. Time/event instant at which packet must become available must be random/programmable. (default: {10})
        num_slots {int} -- If slot based structure is being used for MAC protocol, number of slots. (default: {10})
        sim_end_time {int} -- [description] (default: {10})

    Returns:
        [type] -- [description]
    """
    simTime = list()
    state_id = list(range(0, 3))
    poisson_lambda = sim_end_time / 2
    # state_id values - 0: carrier sense, 1: transmitting, 2: tx end
    events = generate_events(
        num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time)
    return events
    # return perf_metric_1, perf_metric_2, perf_metric_3

# def packet_time_generator(total_time)
# %%


def generate_events(num_nodes=10, num_packets=3, sim_end_time=10):
    poisson_lambda = sim_end_time / 2
    events = np.random.poisson(
        lam=poisson_lambda, size=(num_packets, num_nodes))
    events[events > sim_end_time] = 0
    return events


    # %%
a = simulator()
print(a)
