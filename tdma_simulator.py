# %%
# library imports
import numpy as np
from mac_simulator import *
import matplotlib.pyplot as plt

# initialisations


def sim_int(num_nodes=5, num_packets=5, sim_end_time=1, packet_time=0.01, printFlag=0):

    events = generate_events(num_nodes=num_nodes, num_packets=num_packets,
                             sim_end_time=sim_end_time, event_resolution=0.5*sim_end_time, round=2)
    nodeID = np.zeros((1, events.shape[1]))
    for i in range(events.shape[1]):
        nodeID[0][i] = np.random.randint(num_nodes)

    if(printFlag == 1):
        print(events)

    return events

# %%
# fuctions


def tdma_simulator(events=None, frame_duration=2, num_slots=10, slot_time=0.05, printFlag=0, start_time=0, packet_time=0.01):
    """Function for implementing the event based simulator with TDMA Protocol.

    Keyword Arguments:
        events {numpy array} -- First row - event times, second row - state IDs, third row - packet IDs, fourth row - node IDs
        frame_duration {int} -- Time duration/window size for which the simulator needs to be run
        num_slots {int} -- Number of nodes contending currently in the network. (default: {10})
        slot_time {float} -- This defines the maximum number of packets that can be sent in a slot for each node
        printFlag {int} -- all the debug prints can be enabled with this flag (default: {0})
        start_time {float} -- The global time stamp at which this simulator is being called currently
        packet_time {float} -- Duration of the packet transmission (default: {0.01}) . Unit - seconds
    Returns:
        throughput {float} -- the average throughput for this frame/window duration
        average_latency {float} -- the average latency for this frame/window duration for all the sucessfully transmitted packets
        events_post {numpy array} -- same format as events. This now contains the events which are un-attended to in this TDMA window
    """

    poll_period = slot_time * num_slots
#    print(poll_period)
    no_of_rounds = frame_duration/poll_period
#    print(no_of_rounds)
#    poll_period_intervals = np.linspace(start_time,start_time+frame_duration,no_of_rounds+1)

    slot_start_time = np.zeros((num_slots, int(no_of_rounds)))
    slot_stop_time = np.zeros((num_slots, int(no_of_rounds)))
    for i in range(0, num_slots):
        for j in range(0, int(no_of_rounds)):
            slot_start_time[i, j] = start_time + poll_period*j + slot_time*i
            slot_stop_time[i, j] = start_time + poll_period*j + slot_time*(i+1)
    if(printFlag == 1):
        print('start time array')
        print(slot_start_time)
        print('stop time array')
        print(slot_stop_time)
    final_transmit_time = np.zeros(events.shape[1])
    latency = np.zeros(events.shape[1])
    no_of_packets_sent = 0
    last_packet_sent_time = start_time
    max_packets_per_slot = slot_time/packet_time
    packet_ovr_check = np.ones(
        (num_slots, int(no_of_rounds)))*max_packets_per_slot
    events_pre = events
    events_post = events
    for event_iter in range(events.shape[1]):
        node_id = int(events[3, event_iter])
        for round_iter in range(int(no_of_rounds)):
            if(events[1, event_iter] == 0):
                curr_event_time = events[0, event_iter]
                if(round_iter == 0):
                    min_limit = True
                else:
                    min_limit = curr_event_time >= slot_stop_time[node_id, round_iter-1]
                max_limit = curr_event_time <= (
                    slot_stop_time[node_id, round_iter]-packet_time)
                if(min_limit and max_limit):
                    if(packet_ovr_check[node_id, round_iter] != 0):
                        events[1, event_iter] = 2
                        no_of_packets_sent = no_of_packets_sent+1
                        last_packet_sent_time = slot_stop_time[node_id, round_iter]
    #                    packet_id = events[2,event_iter]
                        final_transmit_time[event_iter] = slot_stop_time[node_id, round_iter]
    #                    print(packet_id)
                        packet_ovr_check[node_id, round_iter] = np.maximum(
                            (packet_ovr_check[node_id, round_iter] - 1), 0)
                    else:
                        events[0, event_iter] = slot_stop_time[node_id,
                                                               round_iter] + packet_time
        if(events[1, event_iter] == 2):
            events_post = np.delete(events_post, 0, axis=1)
            latency[event_iter] = final_transmit_time[event_iter] - \
                events_pre[0, event_iter]
        else:
            latency[event_iter] = 0
#    print(events)

    for post_event_iter in range(events_post.shape[1]):
        if(events_post[0, post_event_iter] <= start_time + frame_duration):
            events_post[0, post_event_iter] = events_post[0,
                                                          post_event_iter] + poll_period

    average_latency = np.mean(latency)
    print(start_time)
    transmission_time = last_packet_sent_time-start_time
    if(transmission_time == 0):
        throughput = 0
    else:
        throughput = no_of_packets_sent/transmission_time

    # print(events_pre)
    # print('final_transmit_time')
    # print(final_transmit_time)
    print('no_of_packets_sent')
    print(no_of_packets_sent)
    print('transmission_time')
    print(transmission_time)
    print('throughput')
    print(throughput)
    print('average_latency')
    print(average_latency)
    print("___________________________")
    return throughput, average_latency, events_post

# test scripts
