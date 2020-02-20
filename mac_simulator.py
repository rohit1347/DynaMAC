# %%
# Add imports
import numpy as np

# %%


def simulator(num_nodes=10, num_packets=3, sim_end_time=10, packet_time=0.01, pflag=0):
    """Function for implementing the event based simulator.

    Keyword Arguments:
        num_nodes {int} -- Number of nodes in the simulation. (default: {10})
        num_packets {int} -- Number of packets at each node. Time/event instant at which packet must become available must be random/programmable. (default: {10})
        sim_end_time {int} -- [description] (default: {10})
        packet_time {float} -- [description] (default: {0.01})
        pflag {int} -- [description] (default: {0})

    Returns:
        [type] -- [description]
    """
    # Asserts
    assert (num_nodes > 0)
    assert num_packets > 0
    assert sim_end_time > 0
    # Variables
    # state_id values - 0: carrier sense, 1: transmitting, 2: tx end
    sent_packets = 0
    total_packets = num_packets * num_nodes
    total_backoff_time = 0
    total_backoffs = 0
    # packet time units - seconds

    # Program

    simEvents = generate_events(
        num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time, event_resolution=10 * packet_time)
    sim_end_check = simEvents[0, :] > sim_end_time
    while (simEvents.shape[1] > 0 and not sim_end_check.all()):
        if pflag:
            print(f'SimEvents={simEvents[1,:]}')
            print(f'SimTime={simEvents[0,:]}')
        curEvent = simEvents[1, 0]
        curTime = simEvents[0, 0]
        if curEvent == 0:
            simEvents = append_event(simEvents, curTime+packet_time, 2)
            simEvents = remove_event(simEvents, 0)
            busy_states = np.logical_and(
                simEvents[0, :] >= curTime, simEvents[0, :] < simEvents[0, -1])
            simEvents[1, busy_states.flatten()] = 1
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
    print(
        f'Simulation has completed. Average latency={latency}s, PSR= {packet_success_ratio} ')
    return latency, packet_success_ratio


# %%


def generate_events(num_nodes=10, num_packets=3, sim_end_time=10, event_resolution=0.5):
    """Generates a single events and states matrix by appending.

    Keyword Arguments:
        num_nodes {int} -- Number of nodes in the simulation (default: {10})
        num_packets {int} -- Number of packets per node (default: {3})
        sim_end_time {int} -- Simulation end time (default: {10})
        event_resolution {float} -- Lambda for exponential function (default: {0.5})

    Returns:
        Numpy array -- Dimensions are [2,num_nodes*num_packets]. First row stores the event times and second row stores the state IDs.
    """
    events = np.random.exponential(
        scale=event_resolution, size=(num_packets, num_nodes))
    events = events.flatten()
    events.sort()
    events = roundoff_events(events)
    events = np.vstack((events, np.zeros(shape=events.shape)))
    # print(f'Events size in gen:{events.shape}')
    return events


def sort_events(events):
    """Sorts the 'events' numpy array according to the event time (first row).

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs

    Returns:
        Numpy array -- Sorted 'events' numpy array
    """
    assert events.shape[0] == 2
    sort_idx = np.argsort(events[0, :])

    events = events[:, sort_idx]
    return events


def generate_backoff(backoff_resolutiton):
    backoff = np.random.exponential(size=(1, 1), scale=backoff_resolutiton)
    backoff = roundoff_events(backoff)
    backoff = backoff[0, 0]
    return backoff


def remove_event(events, idx):
    """Removes event from the event array from a particular column index (event ID).

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs.
        idx {int} -- Event ID which has to be removed.

    Returns:
        Numpy array -- 'Events' numpy array with removed column
    """
    assert events.shape[0] == 2
    events = np.delete(events, idx, axis=1)
    return events


def append_event(events, newTime, newState):
    """Appends new event

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs.
        newTime {float or numpy array} -- Time ID
        newState {int} -- State ID corresponding to the Time ID

    Returns:
        [type] -- [description]
    """
    assert events.shape[0] == 2
    events = np.append(events, np.array([[newTime], [newState]]), axis=1)
    return events


def roundoff_events(events, round=1):
    return np.round(events, decimals=round)


# %%
a = simulator(num_nodes=10, num_packets=100, sim_end_time=1, pflag=1)


# %%
