# %% adding imports
from mac_simulator import *
from tdma_simulator import *
from mac_selection import *
from qlearning_optimized import *
import matplotlib.pyplot as plt


# %%

# simEvents_low = generate_events(
#    num_nodes=num_n_low, num_packets=num_p_low,
#    sim_end_time=duration_till_now+duration_low,
#    round=2,window_size=window_size,start_time = duration_till_now,id_start = pkts_till_now)
#duration_till_now = duration_till_now + duration_low
#simEvents = np.hstack((simEvents,simEvents_low))
#pkts_till_now = simEvents.shape[1]
#
# simEvents_high = generate_events(
#    num_nodes=num_n, num_packets=num_p_high, sim_end_time=duration_high+duration_till_now, round=2,
#    window_size=window_size,start_time = duration_till_now,id_start = pkts_till_now)
#duration_till_now = duration_till_now + duration_high
#simEvents = np.hstack((simEvents,simEvents_high))
#pkts_till_now = simEvents.shape[1]
#
# simEvents_low = generate_events(
#    num_nodes=num_n_low, num_packets=num_p_low,
#    sim_end_time=duration_till_now+duration_low,
#    round=2,window_size=window_size,start_time = duration_till_now,id_start = pkts_till_now)
#duration_till_now = duration_till_now + duration_low
#simEvents = np.hstack((simEvents,simEvents_low))
#pkts_till_now = simEvents.shape[1]
#
# simEvents_high = generate_events(
#    num_nodes=num_n, num_packets=num_p_high, sim_end_time=duration_high+duration_till_now, round=2,
#    window_size=window_size,start_time = duration_till_now,id_start = pkts_till_now)
#duration_till_now = duration_till_now + duration_high
#simEvents = np.hstack((simEvents,simEvents_high))
#pkts_till_now = simEvents.shape[1]


def DynaMAC_comparison(num_n=10, num_p_per_s=2, duration_low=20, duration_high=20, window_size=5, high_low_factor=5):

    num_p_high = int(num_p_per_s*duration_high)
    num_p_low = int(num_p_per_s*duration_low*high_low_factor/2)
    num_n_low = int(num_n/high_low_factor)
    duration_till_now = 0
    pkts_till_now = 0
    simEvents_low = generate_events(
        num_nodes=num_n_low, num_packets=num_p_low,
        sim_end_time=duration_till_now+duration_low,
        round=2, window_size=window_size, start_time=duration_till_now, id_start=pkts_till_now)
    duration_till_now = duration_till_now + duration_low
    #simEvents = np.hstack((simEvents,simEvents_low))
    simEvents = simEvents_low
    pkts_till_now = simEvents.shape[1]

    simEvents_high = generate_events(
        num_nodes=num_n, num_packets=num_p_high, sim_end_time=duration_till_now+duration_high, round=2,
        window_size=window_size, id_start=pkts_till_now, start_time=duration_till_now)
    duration_till_now = duration_till_now + duration_high
    duration = duration_till_now
    print(duration)

    #simEvents = simEvents_high
    simEvents = np.hstack((simEvents, simEvents_high))
    pkts_till_now = simEvents.shape[1]
    latency_final_tdma, xput_final_tdma, prot_arr_tdma, simEvents_post_tdma = DynaMAC_somac_test(
        num_n=num_n, num_p=1, duration=duration, simEvents=simEvents, mac_init=1, somac_en=0, window_size=window_size)

    latency_final_csma, xput_final_csma, prot_arr_csma, simEvents_post_csma = DynaMAC_somac_test(
        num_n=num_n, num_p=1, duration=duration, simEvents=simEvents, mac_init=0, somac_en=0, window_size=window_size)

    latency_final_both, xput_final_both, prot_arr_both, simEvents_post_both = DynaMAC_somac_test(
        num_n=num_n, num_p=1, duration=duration, simEvents=simEvents, mac_init=0,
        somac_en=1, window_size=window_size, opt_type=2, mode=2)

    latency_final_both_s, xput_final_both_s, prot_arr_both_s, simEvents_post_both = DynaMAC_somac_test(
        num_n=num_n, num_p=1, duration=duration, simEvents=simEvents, mac_init=0,
        somac_en=1, window_size=window_size, opt_type=2, mode=3)

    fig7 = plt.figure(num=7, figsize=[9, 6])
    # plt.plot(xput_final_th,'r')
    # plt.plot(xput_final_l,'m')
    plt.plot(xput_final_both, 'k')
    plt.plot(xput_final_both_s, 'r')
    plt.plot(xput_final_tdma, 'b')
    plt.plot(xput_final_csma, 'g')
    plt.xlabel('iteration no', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('throughput (pkts/sec)', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.title('Throughput across number of windows for DynaMAC', fontsize=16)
    plt.legend(['DYNAMAC-E', 'DYNAMAC-S', 'TDMA', 'CSMA'], fontsize=12)
    # plt.legend(['SOMAC-T','TDMA','CSMA'],fontsize=12)
    fig7.show()

    fig8 = plt.figure(num=8, figsize=[9, 6])
    #latency_final[-1] = latency_final[-2]
    # plt.plot(latency_final_th,'r')
    #latency_final_l[-1]= latency_final_l[-2]
    # plt.plot(latency_final_l,'m')
    plt.plot(latency_final_both, 'k')
    plt.plot(latency_final_both_s, 'r')
    plt.plot(latency_final_tdma, 'b')
    plt.plot(latency_final_csma, 'g')
    plt.xlabel('iteration no', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('latency (seconds)', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.title('Latency across number of windows for DynaMAC', fontsize=16)
    plt.legend(['DYNAMAC-E', 'DYNAMAC-S', 'TDMA', 'CSMA'], fontsize=12)
    # plt.legend(['SOMAC-T','TDMA','CSMA'],fontsize=12)
    fig8.show()
# %%
# fig6 = plt.figure(num=6, figsize=[9, 6])
# # plt.plot(prot_arr_th,'r')
# # plt.plot(prot_arr_l,'m')
# plt.plot(prot_arr_both, 'k')
# plt.plot(prot_arr_both_s, 'r')
# plt.plot(prot_arr_tdma, 'b')
# plt.plot(prot_arr_csma, 'g')
# plt.xlabel('iteration no', fontsize=16)
# plt.xticks(fontsize=16)
# plt.ylabel('protocol', fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(True)
# plt.title('Protocol used across windows for DynaMAC', fontsize=16)
# plt.legend(['DYNAMAC-E', 'DYNAMAC-S', 'TDMA', 'CSMA'], fontsize=12)
# fig6.show()
