# %%
# Add imports
from mac_simulator import *
from tdma_simulator import *
# %%


def DynaMAC_switch_test(num_p=5, num_n=5, window_size=10, round=2, duration=200):
    latency_array = np.zeros(1)
    xput_array = np.zeros(1)
    MAC_flag = 0
    latency_tracker = None
    pre_packets = None
    simEvents = generate_events(
        num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=round)
    tp = num_p*num_n
    simEvents_length = simEvents.shape[1]
    for window_start in range(0, duration, window_size):
        if not MAC_flag:
            latency, xput, _, simEvents, latency_tracker, pre_packets = csma_simulator(
                num_nodes=num_n, num_packets=num_p, sim_start_time=window_start, duration=window_size, simEvents=simEvents, latency_tracker=latency_tracker, previously_sent_packets=pre_packets)
        else:
            xput, latency, simEvents = tdma_simulator(
                events=simEvents, frame_duration=window_size, start_time=window_start)
        if simEvents.shape[1] <= simEvents_length:
            MAC_flag = 1
        latency_array = np.append(latency_array, latency)
        xput_array = np.append(xput_array, xput)
    return latency_array, xput_array
