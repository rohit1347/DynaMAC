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
    # Asserts
    assert (num_nodes > 0)
    assert num_packets > 0
    assert num_slots > 0
    assert sim_end_time > 0
    # Variables
    state_id = [0, [0]]
    # state_id values - 0: carrier sense, 1: transmitting, 2: tx end
    packet_time = 5
    # packet time units - seconds

    # Program

    events = generate_events(
        num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time)
    simEvents = update_event_counter(events=simEvents)
    print(f'SimEvents={simEvents}')
    for event_id, event in simEvents:
        if state_id[0] == 0:
            state_id[1] = event+packet_time
            state_id[0] = 1
            simEvents = np.delete(simEvents, event_id)
        elif state_id == 1:
            pass
        elif state_id == 2:
            pass
        else:
            print("Wrong AP state")
        simEvents = update_event_counter(events=simEvents)
    print('Simulation has completed.')
    return events
    # return perf_metric_1, perf_metric_2, perf_metric_3

# def packet_time_generator(total_time)
# %%


def generate_events(num_nodes=10, num_packets=3, sim_end_time=10):
    events = np.random.exponential(
        scale=0.1, size=(num_packets, num_nodes))
    events[events > sim_end_time] = 0
    return events


def update_event_counter(events=generate_events(), end_time=10):
    events = np.unique(events, axis=0)
    events.sort()
    # By putting axis=0, we find unique numbers across each row.
    # Expected to preserve shape of the input.
    return events


def generate_backoff():
    pass

    # %%
a = simulator()
# print(a)


# %%
