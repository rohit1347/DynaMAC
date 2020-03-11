# %%
# Add imports
from mac_simulator import *
from tdma_simulator import *
# %%


def DynaMAC_switch_test(num_p=5, num_n=5, window_size=10, round=2, duration=200):
    latency_array = np.zeros(1)
    xput_array = np.zeros(1)
    MAC_flag = False
    latency_tracker = None
    pre_packets = None
    simEvents = generate_events(
        num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=round)
    print(simEvents)
    tp = num_p*num_n
    simEvents_length = simEvents.shape[1]
    for window_start in range(0, duration, window_size):
        # if simEvents.shape[1] <= simEvents_length/2:
        if window_start >= duration/2:
            MAC_flag = not MAC_flag
        if not MAC_flag:
            latency, xput, _, simEvents, latency_tracker, pre_packets = csma_simulator(
                num_nodes=num_n, num_packets=num_p, sim_start_time=window_start, duration=window_size, simEvents=simEvents, latency_tracker=latency_tracker, previously_sent_packets=pre_packets)
        else:
            print(simEvents.shape[1])
            xput, latency, simEvents = tdma_simulator(
                events=simEvents, frame_duration=window_size, start_time=window_start, num_slots=num_n)

        latency_array = np.append(latency_array, latency)
        xput_array = np.append(xput_array, xput)
    return latency_array, xput_array
