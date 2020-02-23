#%%
# library imports
import numpy as np
from mac_simulator import * 

# initialisations
num_nodes = 5
num_packets = 5
frame_duration = 0.5
start_time = 0
sim_end_time = start_time + frame_duration
events = generate_events(num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time, round=2)
nodeID = np.zeros((1,events.shape[1]))
for i in range(events.shape[1]):
    nodeID[0][i] = np.random.randint(num_nodes)
events = np.append(events,nodeID,axis=0)
print(events)
#%%
# fuctions
def tdma_simulator(events=None , frame_duration = 2 , num_nodes = 10 , slot_time = 0.1 , printFlag=0 , start_time = 0):
    """Function for implementing the event based simulator with TDMA Protocol.

    Keyword Arguments:
        simEvents {numpy array} -- First row - event times, second row - state IDs, third row - packet IDs
        num_nodes {int} -- Number of nodes contending currently in the network. (default: {10})
        packet_time {float} -- Duration of the packet transmission (default: {0.01}) . Unit - seconds
        printFlag {int} -- all the debug prints can be enabled with this flag (default: {0})
    Returns:
        [type] -- [description]
    """
    
    poll_period = slot_time * num_nodes
#    print(poll_period)
    no_of_rounds = frame_duration/poll_period
#    print(no_of_rounds)
#    poll_period_intervals = np.linspace(start_time,start_time+frame_duration,no_of_rounds+1)

    slot_start_time = np.zeros((num_nodes,int(no_of_rounds)))
    slot_stop_time = np.zeros((num_nodes,int(no_of_rounds)))
    for i in range(0,num_nodes):
        for j in range(0,int(no_of_rounds)):
            slot_start_time[i,j] = start_time + poll_period*j + slot_time*i
            slot_stop_time[i,j] = start_time + poll_period*j + slot_time*(i+1)
    print('start time array')
    print(slot_start_time)
    print('stop time array')
    print(slot_stop_time)
    packet_sent_round = np.zeros((int(no_of_rounds),num_nodes))
    latency = np.zeros(events.shape[1])
    for event_iter in range(events.shape[1]):
        node_id = int(events[3,event_iter])
        print('new event')
        for round_iter in range(int(no_of_rounds)):
            curr_event_time = events[0,event_iter]
            if(round_iter==0):
                min_limit = True
            else:
                min_limit = curr_event_time>=slot_stop_time[node_id,round_iter-1]
            max_limit = curr_event_time<=slot_stop_time[node_id,round_iter]
            if(min_limit and max_limit):
                events[1,event_iter] = 2
                packet_id = events[2,event_iter]
                latency[event_iter] = np.maximum((slot_start_time[node_id,round_iter]-curr_event_time),0)
                print(packet_id)
     
    print(packet_sent_round)
    print(events) 
    print(latency)          
            
#    for round_iter in range(0,poll_period_intervals.shape[0]):
#        curr_round_start = poll_period_intervals[round_iter]
#        curr_round_end = curr_round_start + poll_period
#        curr_round_events = np.array([])
##        condition = events[0,:]>=curr_round_start
#        condition=np.logical_and(events[0,:]>=curr_round_start,events[0,:]<=curr_round_end-packet_time)
#        curr_round_events = events[:,condition];
##        print(curr_round_events)
##        print("break here")
#        for i in range(0,num_nodes):
#            find_node_condition = (curr_round_events[3,:]==i)
#            event_node = curr_round_events[:,find_node_condition]
#            print(event_node)
#            print(event_node.shape)
#            print(len(event_node.shape))
#            if(event_node.shape[1]==0):
#                if(event_node[0] <= curr_round_start + packet_time*i):
#                    packet_id = event_node[2]
#                    print('packet is transmitted')
#                    packet_deletion_idx = np.where(events[2]==packet_id)
#                    print(packet_deletion_idx)
#                    np.delete(events,packet_deletion_idx,axis=1)
#            elif(event_node.shape[1]>=1):
#                print("current_node")
#                print(event_node)
#                if(event_node[0,0] <= curr_round_start + packet_time*i):
#                    packet_id = event_node[2,0]
#                    print('packet is transmitted')
#                    packet_deletion_idx = np.where(events[2,:]==packet_id)
#                    print(packet_deletion_idx)
#                    np.delete(events,packet_deletion_idx,axis=1)
#                else:
#                    packet_id = event_node[2,0]
#                    packet_append_idx = np.where(events[2,:]==packet_id)         
#            else:
#                break
#        print(round_iter)
        


# test scripts

tdma_simulator(events=events,num_nodes=num_nodes,frame_duration=1)

# %%
