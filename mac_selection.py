# %%
# Add imports
from mac_simulator import *
from tdma_simulator import *
from qlearning_optimized import *
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
    for iteration, window_start in enumerate(range(0, duration, window_size)):
        simEvents_plot(simEvents, iteration+1, duration=duration, flag=True)
        # https: // seaborn.pydata.org/generated/seaborn.distplot.html
        # if simEvents.shape[1] <= simEvents_length/2:
        if window_start >= duration/2:
            MAC_flag = 1
        if not MAC_flag:
            latency, xput, _, simEvents, latency_tracker, pre_packets = csma_simulator(
                num_nodes=num_n, num_packets=num_p, sim_start_time=window_start, duration=window_size,
                simEvents=simEvents, latency_tracker=latency_tracker, previously_sent_packets=pre_packets)
        # else:
        #     print(simEvents.shape[1])
        #     xput, latency, simEvents = tdma_simulator(
        #         events=simEvents, frame_duration=window_size, start_time=window_start, num_slots=num_n,
        #         slot_time=0.1)

        latency_array = np.append(latency_array, latency)
        xput_array = np.append(xput_array, xput)
    return latency_array, xput_array


def DynaMAC_somac_test(num_p=5, num_n=5, window_size=10, round=2, duration=200):
    latency_array = np.zeros(1)
    xput_array = np.zeros(1)
    MAC_array = np.zeros(1)
    MAC_flag = 0  # 0 - CSMA , 1 - TDMA
    latency_tracker = None
    pre_packets = None
    g_dt = -2
    simEvents = generate_events(
        num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=round)
    simEvents_pre = simEvents
    simEvents_length = simEvents.shape[1]
    for iteration, window_start in enumerate(range(0, duration, window_size)):
        # print(simEvents)
        simEvents_plot(simEvents, iteration+1, duration=duration)
        # https: // seaborn.pydata.org/generated/seaborn.distplot.html
        # if simEvents.shape[1] <= simEvents_length/2:
        if not MAC_flag:
            latency, xput, _, simEvents, latency_tracker, pre_packets = csma_simulator(
                num_nodes=num_n, num_packets=num_p, sim_start_time=window_start, duration=window_size,
                simEvents=simEvents, latency_tracker=latency_tracker, previously_sent_packets=pre_packets)
        else:
            #            print(simEvents.shape[1])
            xput, latency, simEvents = tdma_simulator(
                events=simEvents, frame_duration=window_size, start_time=window_start, num_slots=num_n,
                slot_time=0.1)
        latency_array = np.append(latency_array, latency)
        xput_array = np.append(xput_array, xput)
        decision_class = decision_final(
            xput_array, 1, mode=MAC_flag, g_dt=g_dt)
        # MAC_flag, g_dt = decision_class.result_calc()
        MAC_flag, g_dt = decision_class.decision, decision_class.g_dt
        print(MAC_flag)
        MAC_array = np.append(MAC_array, MAC_flag)

    return latency_array, xput_array, MAC_array, simEvents_pre
