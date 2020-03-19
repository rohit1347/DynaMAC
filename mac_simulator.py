# %%
# Add imports
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    # pkts_sent_in_window_idx = None
    pkts_sent_in_window_idx = list()

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
        previously_sent_packets = list()
        # Latency tracker structure: 1st row- Packet IDs, 2nd row- Start times
        print("Generated new events")
    else:
        total_packets = simEvents.shape[1]
        if not isinstance(latency_tracker, np.ndarray):
            latency_tracker = create_latency_tracker(simEvents)
        if not isinstance(previously_sent_packets, list):
            previously_sent_packets = list()
        print('Using given simEvents')
    # Checking if any packets are starting before the sim window start time
    print(f"Dropped packets:{np.sum(simEvents[0,:]<sim_start_time)}")
    # assert not np.any(
    #     simEvents[0, :] < sim_start_time), f"Some events begin before sim start time"
    assert isinstance(
        latency_tracker, np.ndarray), f"Latency tracker not initialized"
    # print(f"Latency tracker:{latency_tracker}")
    assert isinstance(previously_sent_packets,
                      list), f"Previously sent packets tracker not initialized"
    cond1 = simEvents[0, :] >= sim_start_time
    cond2 = simEvents[0, :] <= sim_end_time
    # pdb.set_trace()
    eligible_packets = np.sum(cond1 & cond2)
    # print(simEvents[2, cond1 & cond2])
    # pdb.set_trace()
    print(f"Num. eligible packets in window: {eligible_packets} packets")

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
            cond1 = simEvents[0, :] >= sim_start_time
            cond2 = simEvents[0, :] <= sim_end_time
            # print(f"ins0 {simEvents[2, cond1 & cond2]}")
            # if not isinstance(pkts_sent_in_window_idx, list):
            # pkts_sent_in_window_idx = list()
            simEvents = append_event(
                simEvents, curTime+packet_time, 2, curID, curNodeID)
            simEvents = remove_event(simEvents, 0)
            cond3 = np.logical_and(
                simEvents[0, :] >= curTime, simEvents[0, :] < curTime+packet_time)
            busy_states = np.logical_and(cond3, simEvents[1, :] != 2)
            simEvents[1, busy_states.flatten()] = 1
            lat_time = latency_tracker[1, np.isclose(
                latency_tracker[0, :], curID)]
            # if lat_time == 0:
            #     pdb.set_trace()
            diff = curTime - lat_time
            # if pflag:
            # print(
            #     f"pid: {curID} diff: {diff}, curTime: {curTime}, lat time: {lat_time}")
            if diff < 0:
                pdb.set_trace()

            latency_tracker[1, np.isclose(
                latency_tracker[0, :], curID)] = diff
            # latency_tracker[1, curID] = curTime - lat_time
            pkt_idx = np.argwhere(np.isclose(
                latency_tracker[0, :], curID).flatten())
            # pkt_idx = latency_tracker[0, (np.isclose(
            #     latency_tracker[0, :], curID).flatten())]
            # pdb.set_trace()
            pkt_idx = pkt_idx.flatten()
            pkt_idx = pkt_idx[0]
            # print(f"pktidx:{pkt_idx}")
            previously_sent_packets.append(pkt_idx)
            pkts_sent_in_window_idx.append(pkt_idx)
            simEvents = sort_events(simEvents)
            sent_packets += 1
            cond1 = simEvents[0, :] >= sim_start_time
            cond2 = simEvents[0, :] <= sim_end_time
            # print(f"ins0post {simEvents[2, cond1 & cond2]}")

        elif curState == 1:
            cond1 = simEvents[0, :] >= sim_start_time
            cond2 = simEvents[0, :] <= sim_end_time
            # print(f"ins1 {simEvents[2, cond1 & cond2]}")
            new_state_added_flag = 0
            backoff = 0
            endState = simEvents[1, :] == 2
            if sum(endState) != 1:
                endState = np.where(simEvents[1, :] == 0)[0]
                # true_states = np.where(endState)
                endState = endState[0]

            # assert np.sum(endState) <= 1, f"Endstate error: {endState}"
            endTime = simEvents[0, endState]
            # try:
            #     # assert endTime.shape[0] == 1 or endTime.shape == ()
            #     assert endTime.shape[0] == 1 or endTime.shape == ()
            # except:
            #     pdb.set_trace()
            num_backoff = 0
            while new_state_added_flag < 1:
                backoff += generate_backoff(2*packet_time,
                                            round=round, num_backoff=num_backoff, max_backoff=5)
                newTime = curTime + backoff
                num_backoff += 1
                if pflag:
                    print(f"curID:{curID},backoff:{backoff}")
                end_time_check = endTime < newTime
                # end_time_check = end_time_check[0]
                if end_time_check:
                    simEvents = append_event(
                        simEvents, newTime, 0, curID, curNodeID)
                    simEvents = sort_events(simEvents)
                    new_state_added_flag += 1
                backoff += 1
            total_backoff_time += backoff
            total_backoffs += 1
            simEvents = remove_event(simEvents, 0)
            assert np.unique(simEvents[2, :]).shape == simEvents[2, :].shape
            cond1 = simEvents[0, :] >= sim_start_time
            cond2 = simEvents[0, :] <= sim_end_time
            # print(f"ins1post {simEvents[2, cond1 & cond2]}")

        elif curState == 2:
            cond1 = simEvents[0, :] >= sim_start_time
            cond2 = simEvents[0, :] <= sim_end_time
            # print(f"ins2 {simEvents[2, cond1 & cond2]}")
            if simEvents.shape[1] == 1:
                tx_end_time = simEvents[0]
            else:
                tx_end_time = None
            simEvents = remove_event(simEvents, 0)
            cond1 = simEvents[0, :] >= sim_start_time
            cond2 = simEvents[0, :] <= sim_end_time
            # print(f"ins2post {simEvents[2, cond1 & cond2]}")

        sim_end_check = simEvents[0, :] > sim_end_time
    if isinstance(pkts_sent_in_window_idx, list):
        latency = np.mean(latency_tracker[1, np.asarray(
            pkts_sent_in_window_idx, dtype=int)])
        if latency > 2:
            # pdb.set_trace()
            print(f"Latency>2")
    else:
        latency = np.inf
    # print(
    #     f"pre_track{latency_tracker[1,np.asarray(pkts_sent_in_window_idx,dtype=int)]}")
    # pdb.set_trace()
    global_latency = np.mean(
        latency_tracker[1, np.asarray(previously_sent_packets, dtype=int)])
    packet_success_ratio = sent_packets / eligible_packets
    xput = sent_packets / duration
    print(f"Tback:{total_backoff_time}")
    print(
        f'Sim window stats - average latency={latency}s, PSR= {packet_success_ratio}, Ineligible packets: {total_packets-eligible_packets}, Tx End Time: {tx_end_time}')
    print(
        f"Window xput: {sent_packets/duration} packets/second, sent packets:{sent_packets}")
    if pflag:
        print(f"SimEvents: {simEvents}")
    print("---------------------------------------")
    print(
        f"Mean latency across windows: {global_latency}")
    return latency, xput, tx_end_time, simEvents, latency_tracker, previously_sent_packets


# %%


def generate_events(num_nodes=10, num_packets=3, sim_end_time=10, round=1, window_size=10):
    """Generates a single events and states matrix by appending.
    Keyword Arguments:
        num_nodes {int} -- Number of nodes in the simulation (default: {10})
        num_packets {int} -- Number of packets per node (default: {3})
        sim_end_time {int} -- Simulation end time (default: {10})
        event_resolution {float} -- Lambda for exponential function (default: {0.5})
    Returns:
        Num_py array -- Dimensions are [4,num_nodes*num_packets]. First row stores the event times, second row stores the state IDs, third row stores packet IDs, fourth row stores node IDs.
    """
    events = np.zeros((num_packets, num_nodes))
    for node in range(num_nodes):
        if(node % 4 == 0):
            events[:, node] = np.random.uniform(
                0, 0.2*sim_end_time, num_packets)
        elif(node % 4 == 1):
            events[:, node] = np.random.uniform(
                0.8*sim_end_time, sim_end_time, num_packets)
        elif(node % 4 == 2):
            events[:, node] = np.random.normal(
                0.5*sim_end_time, 5*window_size, num_packets)
        else:
            events[:, node] = np.random.random(num_packets)*sim_end_time
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

# %%


def sort_events(events):
    """Sorts the 'events' numpy array according to the event time (first row).

    Arguments:
        events {numpy array} -- First row - event times, second row - state IDs

    Returns:
        Num_py array -- Sorted 'events' numpy array
    """
    assert events.shape[0] == 4
    sort_idx = np.argsort(events[0, :], kind='stable')

    events = events[:, sort_idx]
    return events


def generate_backoff(backoff_resolution, round=1, num_backoff=1, max_backoff=5):
    backoff = 0
    num_backoff = np.minimum(num_backoff, max_backoff)
    while backoff == 0:
        backoff = np.random.uniform(
            low=num_backoff*2, high=num_backoff*2+2, size=(1, 1))
        # backoff = np.random.exponential(
        #     size=(1, 1), scale=num_backoff*backoff_resolution)
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
        simEvents = generate_events(
            num_nodes=num_n, num_packets=num_p, sim_end_time=duration, window_size=duration/10)
        latency_tracker = None
        prev_packets = None
        # Initialzing a random simEvents for while loop to start
    else:
        i = 0
        tp = simEvents.shape[1]
    while simEvents.shape[1] > 0:
        print(f"simEvents length: {simEvents.shape[1]}")
        # if i == 0:
        #     latency, xput, tx_end_time, simEvents, latency_tracker, prev_packets = csma_simulator(
        #         num_nodes=num_n, num_packets=num_p, sim_start_time=i, duration=duration, pflag=0, round=2, packet_time=packet_time)
        #     i += 1
        # else:
        latency, xput, tx_end_time, simEvents, latency_tracker, prev_packets = csma_simulator(
            simEvents=simEvents, sim_start_time=i*duration, duration=duration, pflag=0, latency_tracker=latency_tracker, packet_time=packet_time, previously_sent_packets=prev_packets)
        i += 1
    global_xput = xput
    global_latency = np.mean(
        latency_tracker[1, np.asarray(prev_packets, dtype=int)])
    print(f"Throughput={xput} packets/second")
    return tp, global_xput, global_latency, latency_tracker


def create_latency_tracker(simEvents):
    total_packets = simEvents.shape[1]
    latency_tracker = np.zeros((2, total_packets))
    latency_tracker[0, :] = simEvents[2, :].astype(int)
    latency_tracker[1, :] = simEvents[0, :]
    return latency_tracker
# %% Generating plots function


def simEvents_plot(simEvents, iteration, duration=0, flag=True):
    # Filtering to obtain left over packets (state = 0)
    if flag:
        simEvents = simEvents[:, np.isclose(simEvents[1, :], 0)]
        plt.figure(figsize=[9, 6])
        sns.distplot(simEvents[0, :])
        plt.axvline(x=duration)
        plt.xlabel('Packet Timestamps')
        plt.ylabel('Probability')
        plt.title(f"Window: {iteration}")
