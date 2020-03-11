# %%
# Add imports
import numpy as np
import pdb
import matplotlib.pyplot as plt

# %%


def csma_simulator(num_nodes=10, num_packets=3, sim_start_time=0, duration=10, packet_time=0.01, pflag=0, simEvents=None, round=1, latency_tracker=None, previously_sent_packets=None):
    """Function for implementing the event based simulator.

    Keyword Arguments:
        num_nodes {int} -- Number of nodes in the simulation. (default: {10})
        num_packets {int} -- Number of packets at each node. Time/event instant at which packet must become available must be random/programmable. (default: {10})
        sim_start_time {int} -- [description] (default: {0})
        duration {int} -- [description] (default: {10})
        packet_time {float} -- [description] (default: {0.01})
        pflag {int} -- [description] (default: {0})
        simEvents {numpy array} -- First row - event times, second row - state IDs, third row - packet IDs
        round {int} -- [description] (default: {1})
        latency_tracker {[type]} -- [description] (default: {None})
        previously_sent_packets {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    # Asserts
    assert (num_nodes > 0)
    assert num_packets > 0
    assert duration > 0

    # Variables
    # state_id values - 0: carrier sense, 1: transmitting, 2: tx end
    sim_end_time = sim_start_time+duration
    sent_packets = 0
    total_backoff_time = 0
    total_backoffs = 0
    pkts_sent_in_window_idx = None

    # packet time units - seconds
    # ADD - per user latency tracking
    # ADD - sim start time and duration
    tx_end_time = None
    # Program
    if not isinstance(simEvents, np.ndarray):
        simEvents = generate_events(
            num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time, round=round)
        total_packets = num_packets * num_nodes
        latency_tracker = create_latency_tracker(simEvents)
        print(f"Initial latency tracker: {latency_tracker}")
        previously_sent_packets = np.empty(1, dtype=int)
        # Latency tracker structure: 1st row- Packet IDs, 2nd row- Start times
        print("Generated new events")
    else:
        total_packets = simEvents.shape[1]
        if not isinstance(latency_tracker, np.ndarray):
            latency_tracker = create_latency_tracker(simEvents)
        if not isinstance(previously_sent_packets, np.ndarray):
            previously_sent_packets = np.empty(1, dtype=int)
        # simEvents = rezero_indices(simEvents)
        print('Using given simEvents')
    # Checking if any packets are starting before the sim window start time
    if np.any(simEvents[0, :] < sim_start_time):
        # pdb.set_trace()
        print(f"Some events begin before sim start time")
    # assert not np.any(
        # simEvents[0, :] < sim_start_time), f"Some events begin before sim start time"
    assert isinstance(
        latency_tracker, np.ndarray), f"Latency tracker not initialized"
    assert isinstance(previously_sent_packets,
                      np.ndarray), f"Previously sent packets tracker not initialized"
    cond1 = simEvents[0, :] >= sim_start_time
    cond2 = simEvents[0, :] <= sim_end_time
    # pdb.set_trace()
    eligible_packets = np.sum(cond1 & cond2)
    print("Num. eligible packets in window: {eligible_packets} packets")

    print(
        f"Num. ineligible packets at simulation start: {total_packets-eligible_packets}")

    sim_end_check = simEvents[0, :] > sim_end_time
    while (simEvents.shape[1] > 0 and not sim_end_check.all()):
        if pflag:
            print("------------------------------")
            print(f"Packet IDs={simEvents[2,:]}")
            print(f'SimStates={simEvents[1,:]}')
            print(f'SimTime={simEvents[0,:]}')
            print("------------------------------")
        curTime = simEvents[0, 0]
        curState = simEvents[1, 0]
        curID = simEvents[2, 0].astype(int)
        curNodeID = simEvents[3, 0]
        if curState == 0:
            if not isinstance(pkts_sent_in_window_idx, np.ndarray):
                pkts_sent_in_window_idx = np.zeros(1, dtype=int)
            simEvents = append_event(
                simEvents, curTime+packet_time, 2, curID, curNodeID)
            simEvents = remove_event(simEvents, 0)
            busy_states = np.logical_and(
                simEvents[0, :] >= curTime, simEvents[0, :] < simEvents[0, -1])
            simEvents[1, busy_states.flatten()] = 1
            lat_time = latency_tracker[1, np.isclose(
                latency_tracker[0, :], curID)]
            # lat_time = latency_tracker[1, curID]
            # pdb.set_trace()
            diff = curTime-lat_time
            print(
                f"pid: {curID} diff: {diff}, curTime: {curTime}, lat time: {lat_time}")

            if diff < 0:
                pdb.set_trace()

            # latency_tracker[1, np.isclose(
            #     latency_tracker[0, :], curID)] = curTime - lat_time
            latency_tracker[1, curID] = curTime - lat_time
            # print(f"Latency tracker:{latency_tracker}")
            # previously_sent_packets = np.append(previously_sent_packets, np.argwhere(np.isclose(
            #     latency_tracker[0, :], curID)))
            previously_sent_packets = np.append(
                previously_sent_packets, curID)
            # pkts_sent_in_window_idx = np.append(pkts_sent_in_window_idx, np.argwhere(np.isclose(
            #     latency_tracker[0, :], curID)))
            pkts_sent_in_window_idx = np.append(pkts_sent_in_window_idx, curID)
            simEvents = sort_events(simEvents)
            sent_packets += 1

        elif curState == 1:
            new_state_added_flag = 0
            backoff = 0
            endState = simEvents[1, :] == 2
            assert np.sum(endState) == 1, f"Endstate error: {endState}"
            endTime = simEvents[0, endState]
            num_backoff = 1
            while new_state_added_flag < 1:
                backoff += generate_backoff(2*packet_time,
                                            round=round, num_backoff=num_backoff, max_backoff=5)
                newTime = curTime + backoff
                if pflag:
                    print(f"curID:{curID},backoff:{backoff}")
                end_time_check = endTime < newTime
                if end_time_check:
                    simEvents = append_event(
                        simEvents, newTime, 0, curID, curNodeID)
                    simEvents = sort_events(simEvents)
                    new_state_added_flag += 1
                backoff += 1
            total_backoff_time += backoff
            total_backoffs += 1
            simEvents = remove_event(simEvents, 0)

        elif curState == 2:
            if simEvents.shape[1] == 1:
                tx_end_time = simEvents[0]
            else:
                tx_end_time = None
            simEvents = remove_event(simEvents, 0)

        sim_end_check = simEvents[0, :] > sim_end_time
    if isinstance(pkts_sent_in_window_idx, np.ndarray):
        latency = np.mean(latency_tracker[1, pkts_sent_in_window_idx])
    else:
        latency = np.inf

    global_latency = np.mean(latency_tracker[1, previously_sent_packets])
    packet_success_ratio = sent_packets / eligible_packets
    xput = sent_packets/duration
    print(
        f'Sim window stats - average latency={latency}s, PSR= {packet_success_ratio}, Ineligible packets: {total_packets-eligible_packets}, Tx End Time: {tx_end_time}')
    print(f"Window xput: {sent_packets/duration} packets/second")
    if pflag:
        print(f"SimEvents: {simEvents}")
    print("---------------------------------------")
    print(
        f"Mean latency across windows: {global_latency}")
    return latency, xput, tx_end_time, simEvents, latency_tracker, previously_sent_packets


# %%


def generate_events(num_nodes=10, num_packets=3, sim_end_time=10, round=1):
    """Generates a single events and states matrix by appending.

    Keyword Arguments:
        num_nodes {int} -- Number of nodes in the simulation (default: {10})
        num_packets {int} -- Number of packets per node (default: {3})
        sim_end_time {int} -- Simulation end time (default: {10})
        event_resolution {float} -- Lambda for exponential function (default: {0.5})

    Returns:
        Num_py array -- Dimensions are [3,num_nodes*num_packets]. First row stores the event times, second row stores the state IDs, third row stores packet IDs, fourth row stores node IDs.
    """
    event_resolution = 0.6*sim_end_time
    events = np.zeros((num_packets, num_nodes))
    for node in range(num_nodes):
        events[:, node] = np.random.exponential(
            scale=event_resolution, size=num_packets)
    events = events.flatten()
    num_events = events.shape[0]
    # events.sort()
    events = roundoff_events(events, round=round)
    events = np.vstack((events, np.zeros(shape=events.shape)))
    events = np.vstack((events, np.arange(num_events, dtype=int)))
    events = np.vstack((events, np.repeat([range(num_nodes)], num_packets)))
    events = sort_events(events)
    print(f'Events size in gen:{events.shape}')
    # ADD node id in the fourth row
    return events


def sort_events(events):
    """Sorts the 'events' numpy array according to the event time (first row).

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs

    Returns:
        Num_py array -- Sorted 'events' numpy array
    """
    assert events.shape[0] == 4
    sort_idx = np.argsort(events[0, :])

    events = events[:, sort_idx]
    return events


def generate_backoff(backoff_resolution, round=1, num_backoff=1, max_backoff=10):
    backoff = 0
    num_backoff = np.minimum(num_backoff, max_backoff)
    while backoff == 0:
        backoff = np.random.exponential(
            size=(1, 1), scale=num_backoff*backoff_resolution)
        backoff = roundoff_events(backoff, round=round)
        backoff = backoff[0, 0]
    return backoff


def remove_event(events, idx):
    """Removes event from the event array from a particular column index (event ID).

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs.
        idx {int} -- Event ID which has to be removed.

    Returns:
        Num_py array -- 'Events' numpy array with removed column
    """
    assert events.shape[0] == 4
    events = np.delete(events, idx, axis=1)
    return events


def append_event(events, newTime, newState, newID, newNodeID):
    """Appends new event

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs.
        newTime {float or numpy array} -- Time ID
        newState {int} -- State ID corresponding to the Time ID

    Returns:
        [type] -- [description]
    """
    assert events.shape[0] == 4
    events = np.append(events, np.array(
        [[newTime], [newState], [newID], [newNodeID]]), axis=1)
    return events


def roundoff_events(events, round=1):
    return np.round(events, decimals=round)


def rezero_indices(events):
    num_packets = events.shape[1]
    new_packet_ids = np.arange(num_packets)
    old_packet_ids = events[2, :]
    old_sort_idx = np.argsort(old_packet_ids)
    new_packet_ids = new_packet_ids[old_sort_idx]
    events[2, :] = new_packet_ids
    return events


# %%
def CSMA_simulator(num_p=5, num_n=5, duration=1, packet_time=0.01, simEvents=None):
    if not isinstance(simEvents, np.ndarray):
        i = 0
        tp = num_p*num_n
        simEvents = np.zeros(shape=(4, tp))
        # Initialzing a random simEvents for while loop to start
    else:
        i = 0
        tp = simEvents.shape[1]
    while simEvents.shape[1] > 0:
        print(f"simEvents length: {simEvents.shape[1]}")
        if i == 0:
            latency, xput, tx_end_time, simEvents, latency_tracker, prev_packets = csma_simulator(
                num_nodes=num_n, num_packets=num_p, sim_start_time=i, duration=duration, pflag=0, round=2, packet_time=packet_time)
            i += 1
        else:
            latency, xput, tx_end_time, simEvents, latency_tracker, prev_packets = csma_simulator(
                simEvents=simEvents, sim_start_time=i*duration, duration=duration, pflag=0, latency_tracker=latency_tracker, packet_time=packet_time, previously_sent_packets=prev_packets)
            i += 1
    global_xput = (tp/tx_end_time)[0]
    print(f"Throughput={xput} packets/second")
    return tp, global_xput, latency, latency_tracker


def create_latency_tracker(simEvents):
    total_packets = simEvents.shape[1]
    latency_tracker = np.zeros((2, total_packets))
    latency_tracker[0, :] = simEvents[2, :].astype(int)
    latency_tracker[1, :] = simEvents[0, :]
    return latency_tracker
# %% Generating plots function


def generate_xput_plots(num_p=5, num_n_start=5, num_n_delta=5, num_n_end=50, montecarlo=1, duration=1):
    assert montecarlo >= 1, 'Number of montecarlo runs must be atleast 1'
    num_ns = range(num_n_start, num_n_end, num_n_delta)
    xputs = np.zeros(shape=(montecarlo, len(num_ns)))
    total_packets = np.zeros(shape=(montecarlo, len(num_ns)))
    mc_latency = np.zeros(shape=(montecarlo, len(num_ns)))
    for it in range(0, montecarlo):
        for count, num_n in enumerate(num_ns):
            tp, xput, latency, _ = CSMA_simulator(
                num_p=5, num_n=num_n, duration=duration)
            total_packets[it, count] = tp
            xputs[it, count] = xput
            mc_latency[it, count] = latency
    xputs_mean = np.mean(xputs, axis=0)
    total_packets_mean = np.mean(total_packets, axis=0)
    xputs_var = np.var(xputs, axis=0)
    latency_mean = np.mean(mc_latency, axis=0)
    latency_var = np.var(mc_latency, axis=0)
    fig1 = plt.figure(num=1, figsize=[9, 6])
    plt.errorbar(num_ns, xputs_mean,
                 yerr=xputs_var, label='Throughput')
    plt.fill_between(num_ns, xputs_mean-xputs_var,
                     xputs_mean+xputs_var, alpha=0.5)
    plt.grid(True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Throughput (pkt/sec)')
    plt.legend(loc='lower right')
    plt.title('CSMA: Throughput vs Number of Nodes')
    fig1.show()
    print(latency_mean, latency_var)
    fig2 = plt.figure(num=2, figsize=[9, 6])
    plt.errorbar(num_ns, latency_mean,
                 yerr=latency_var, label='Latency', color='red')
    plt.fill_between(num_ns, latency_mean-latency_var,
                     latency_mean+latency_var, alpha=0.5, facecolor='r')
    plt.grid(True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Latency/packet (s)')
    plt.legend(loc='lower right')
    plt.title('CSMA: Average Latency vs Number of Nodes')
    fig2.show()
