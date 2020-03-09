#%%
# library imports
import numpy as np
from mac_simulator import * 
import matplotlib.pyplot as plt
    
# initialisations
def sim_int(num_nodes=5, num_packets=5,sim_end_time=1,packet_time=0.01,printFlag=0):

    events = generate_events(num_nodes=num_nodes, num_packets=num_packets, sim_end_time=sim_end_time, event_resolution=0.5*sim_end_time, round=2)
    nodeID = np.zeros((1,events.shape[1]))
    for i in range(events.shape[1]):
        nodeID[0][i] = np.random.randint(num_nodes)

    if(printFlag==1):
        print(events)
        
    return events
    
#%%
# fuctions
def tdma_simulator(events=None , frame_duration = 2 , num_slots = 10 , slot_time = 0.05 , printFlag=0 , start_time = 0,packet_time=0.01):
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

    slot_start_time = np.zeros((num_slots,int(no_of_rounds)))
    slot_stop_time = np.zeros((num_slots,int(no_of_rounds)))
    for i in range(0,num_slots):
        for j in range(0,int(no_of_rounds)):
            slot_start_time[i,j] = start_time + poll_period*j + slot_time*i
            slot_stop_time[i,j] = start_time + poll_period*j + slot_time*(i+1)
    if(printFlag==1):
        print('start time array')
        print(slot_start_time)
        print('stop time array')
        print(slot_stop_time)
    final_transmit_time = np.zeros(events.shape[1])
    latency = np.zeros(events.shape[1])
    no_of_packets_sent = 0
    last_packet_sent_time = start_time;
    max_packets_per_slot = slot_time/packet_time
    packet_ovr_check = np.ones((num_slots,int(no_of_rounds)))*max_packets_per_slot
    events_pre = events
    events_post = events
    for event_iter in range(events.shape[1]):
        node_id = int(events[3,event_iter])
        for round_iter in range(int(no_of_rounds)):
            if(events[1,event_iter]==0):
                curr_event_time = events[0,event_iter]
                if(round_iter==0):
                    min_limit = True
                else:
                    min_limit = curr_event_time>=slot_stop_time[node_id,round_iter-1]
                max_limit = curr_event_time<=(slot_stop_time[node_id,round_iter]-packet_time)
                if(min_limit and max_limit):
                    if(packet_ovr_check[node_id,round_iter]!=0):
                        events[1,event_iter] = 2
                        no_of_packets_sent = no_of_packets_sent+1
                        last_packet_sent_time = start_time + slot_stop_time[node_id,round_iter]
    #                    packet_id = events[2,event_iter]
                        final_transmit_time[event_iter] = slot_stop_time[node_id,round_iter]
    #                    print(packet_id)
                        packet_ovr_check[node_id,round_iter] = np.maximum((packet_ovr_check[node_id,round_iter] - 1),0)
                    else:
                        events[0,event_iter] = slot_stop_time[node_id,round_iter] + packet_time
        if(events[1,event_iter] == 2):
            events_post = np.delete(events_post, 0, axis=1)
            latency[event_iter] = final_transmit_time[event_iter] - events_pre[0,event_iter]
        else:
            latency[event_iter] = 0
#    print(events) 
    
    average_latency = np.mean(latency(idx))
    transmission_time = last_packet_sent_time-start_time
    if(transmission_time == 0):
        throughput = 0
    else:
        throughput = no_of_packets_sent/transmission_time
    
    if(printFlag==1):
        print(events_pre) 
        print('final_transmit_time')
        print(final_transmit_time)
        print('no_of_packets_sent')
        print(no_of_packets_sent)
        print('transmission_time')
        print(transmission_time)
        print('throughput')
        print(throughput)
        print('average_latency')
        print(average_latency)
    
    return throughput,average_latency,events_post
           
# test scripts
    
#%%
num_packets = 5
frame_duration = 1
start_time = 0
packet_time = 0.01
num_nodes = 5
sim_end_time = start_time + frame_duration
events = sim_int(num_nodes=num_nodes,num_packets=num_packets,sim_end_time=sim_end_time,packet_time=packet_time,printFlag=1)
throughput,avg_latency,post_events = tdma_simulator(events=events,num_slots=num_nodes,frame_duration=frame_duration,slot_time=5*packet_time,printFlag=1)
    

#%%
num_packets = 20
frame_duration = 5
start_time = 0
packet_time = 0.01
num_slots = 20
num_sims = 1
throughput = np.zeros((num_slots,num_sims))
avg_latency = np.zeros((num_slots,num_sims))
num_nodes = np.linspace(1,num_slots,num_slots)
sim_end_time = start_time + frame_duration
for sim_node_iter in range(num_slots):
    for mont_iter in range(num_sims):
        events = sim_int(num_nodes=int(num_nodes[sim_node_iter]),num_packets=num_packets,sim_end_time=sim_end_time,packet_time=packet_time,printFlag=0)
        throughput[sim_node_iter,mont_iter],avg_latency[sim_node_iter,mont_iter] = tdma_simulator(events=events,num_slots=num_slots,frame_duration=frame_duration,slot_time=5*packet_time,printFlag=0)

avg_th_user = np.mean(throughput,axis=1)
var_th_user = np.var(throughput,axis=1)
avg_latency_user = np.mean(avg_latency,axis=1)
var_latency_user = np.var(avg_latency,axis=1)
print(avg_th_user)
print(var_th_user)
print(avg_latency_user)
print(var_latency_user)
fig = plt.figure(num=1)
#plt.plot(num_nodes,th_user)
plt.errorbar(num_nodes, avg_th_user,
             yerr=var_th_user, label='Throughput')
plt.fill_between(num_nodes, avg_th_user-var_th_user,
                 avg_th_user+var_th_user, alpha=0.5)
plt.xlabel('Number of Nodes')
plt.ylabel('Throughput (pkt/sec)')
plt.legend(loc='lower right')
plt.title('TDMA: Throughput vs Number of Nodes')
plt.grid(True)
plt.show()

fig = plt.figure(num=2)
plt.errorbar(num_nodes, avg_latency_user,
             yerr=var_latency_user, label='Latency',color='red')
plt.fill_between(num_nodes, avg_latency_user-var_latency_user,
                 avg_latency_user+var_latency_user, facecolor='r', alpha=0.5)
plt.xlabel('Number of Nodes')
plt.ylabel('latency (seconds)')
plt.legend(loc='lower right')
plt.title('TDMA: Latency vs Number of Nodes')
plt.grid(True)
plt.show()
# %%
#
#fig = plt.figure(num=1)
#    plt.errorbar(total_packets_mean, xputs_mean,
#                 yerr=xputs_var, label='Throughput')
#    plt.fill_between(total_packets_mean, xputs_mean-xputs_var,
#                     xputs_mean+xputs_var, alpha=0.5)
#    plt.grid(True)
#    plt.xlabel('Total packets')
#    plt.ylabel('Throughput (pkt/sec)')
#    plt.legend(loc='lower right')
#    plt.title('CSMA: Throughput vs Number of Packets')

# %%

# combining the tdma and csma simulator

# step 1 : generate events across 500 seconds
num_packets = 100
sim_duration = 40.0
frame_duration = 2.0
start_time = 0
packet_time = 0.01
num_nodes = 10
sim_end_time = start_time + sim_duration
events = sim_int(num_nodes=num_nodes,num_packets=num_packets,sim_end_time=sim_end_time,packet_time=packet_time,printFlag=0)

# step 2 : each csma and tdma window can be 10 seconds
num_frames = int(sim_duration/frame_duration)
throughput = np.zeros(num_frames)
avg_latency = np.zeros(num_frames)
pre_events = events
for frame_iter in range(num_frames):
    start_time_frame = start_time + frame_iter*frame_duration
    throughput[frame_iter],avg_latency[frame_iter],events = tdma_simulator(events=events,num_slots=num_nodes,frame_duration=frame_duration,start_time = start_time_frame,slot_time=2*packet_time,printFlag=0)

#print(events)
print(throughput)
print(avg_latency)
print(pre_events[0])
