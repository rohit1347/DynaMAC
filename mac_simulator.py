# %%
# Add imports
import numpy as np
import pdb
# %%


def simulator(num_nodes=10, num_packets=3, num_slots=10, sim_end_time=10, packet_time=0.01):
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
    # state_id values - 0: carrier sense, 1: transmitting, 2: tx end
    packet_time = packet_time
    sent_packets = 0
    total_packets = num_packets * num_slots
    total_backoff_time = 0
    total_backoffs = 0
    # packet time units - seconds

    # Program

    simEvents = generate_events(
        num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time, event_resolution=10 * packet_time)
    sim_end_check = simEvents[0, :] > sim_end_time
    while (simEvents.shape[1] > 0 and not sim_end_check.all()):
        print(f'SimEvents={simEvents}')
        curEvent = simEvents[1, 0]
        curTime = simEvents[0, 0]
        if curEvent == 0:
            simEvents = append_event(simEvents, curTime+packet_time, 2)
            simEvents = remove_event(simEvents, 0)

            busy_states = np.logical_and(
                simEvents[0, :] >= curTime, simEvents[0, :] < simEvents[0, -1])
            simEvents[1, busy_states] = 1
            simEvents = sort_events(simEvents)
            sent_packets += 1
        elif curEvent == 1:
            new_state_added_flag = 0
            backoff = 0
            endState = simEvents[1, :] == 2
            assert np.sum(endState) == 1
            endTime = simEvents[0, endState]
            while new_state_added_flag < 1:
                backoff += generate_backoff(10 * packet_time)
                newTime = curTime + backoff
                pdb.set_trace()
                end_time_check = endTime < newTime
                if end_time_check:
                    simEvents = append_event(simEvents, newTime, 0)
                    simEvents = sort_events(simEvents)
                    new_state_added_flag += 1
                else:
                    backoff += backoff
            total_backoff_time += backoff
            total_backoffs += 1
            simEvents = remove_event(simEvents, 0)
        elif curEvent == 2:
            simEvents = remove_event(simEvents, 0)
        sim_end_check = simEvents[0, :] > sim_end_time
    latency = total_backoff_time / total_backoffs
    packet_success_ratio = sent_packets/total_packets
    print('Simulation has completed.')
    return latency, packet_success_ratio

# def packet_time_generator(total_time)
# %%


def generate_events(num_nodes=10, num_packets=3, sim_end_time=10, event_resolution=0.5):
    events = np.random.exponential(
        scale=event_resolution, size=(num_packets, num_nodes))
    events = events.flatten()
    events.sort()
    events = np.vstack((events, np.zeros(shape=events.shape)))
    # print(f'Events size in gen:{events.shape}')
    return events


def sort_events(events):
    assert events.shape[0] == 2
    events[0, :] = roundoff_events(events[0, :])
    sort_idx = np.argsort(events[0, :])
    events = events[:, sort_idx]
    return events


def generate_backoff(backoff_resolutiton):
    backoff = np.random.exponential(size=(1, 1), scale=backoff_resolutiton)
    backoff = roundoff_events(backoff)
    return backoff


def remove_event(events, idx):
    assert events.shape[0] == 2
    events = np.delete(events, idx, axis=1)
    return events


def append_event(events, newTime, newState):
    assert events.shape[0] == 2
    events = np.append(events, np.array([[newTime], [newState]]), axis=1)
    return events


def roundoff_events(events, round=1):
    return np.round(events, decimals=round)


# %%
a = simulator()
# print(a)


# %%
