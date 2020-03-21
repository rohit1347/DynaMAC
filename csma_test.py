# %% adding imports
from mac_simulator import *
from tdma_simulator import *
from mac_selection import *
from qlearning_optimized import *
import matplotlib.pyplot as plt
# # %%
# num_n = 10
# num_p = 2000
# duration = 1000
# window_size = 20
# simEvents = generate_events(
#     num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=2, window_size=window_size)

# latency_final_tdma, xput_final_tdma, prot_arr_tdma, simEvents_post_tdma = DynaMAC_somac_test(
#     num_n=num_n, num_p=num_p, duration=duration, simEvents=simEvents, mac_init=1, somac_en=0, window_size=window_size)

# latency_final_csma, xput_final_csma, prot_arr_csma, simEvents_post_csma = DynaMAC_somac_test(
#     num_n=num_n, num_p=num_p, duration=duration, simEvents=simEvents, mac_init=0, somac_en=0, window_size=window_size)

# latency_final, xput_final, prot_arr, simEvents_post = DynaMAC_somac_test(
#     num_n=num_n, num_p=num_p, duration=duration, simEvents=simEvents, mac_init=0, somac_en=1, window_size=window_size)
# # %%
# plt.figure()
# plt.plot(prot_arr, 'r')
# plt.plot(prot_arr_tdma, 'b')
# plt.plot(prot_arr_csma, 'g')
# plt.show()
# plt.figure()
# plt.plot(xput_final, 'r')
# plt.plot(xput_final_tdma, 'b')
# plt.plot(xput_final_csma, 'g')
# plt.show()
# plt.figure()
# plt.plot(latency_final, 'r')
# plt.plot(latency_final_tdma, 'b')
# plt.plot(latency_final_csma, 'g')
# plt.show()
# %%
# latency_final, xput_final = DynaMAC_switch_test(
#    num_n=30, num_p=40, duration=200)
# %%
#_, xput, _, lat = CSMA_simulator(duration=500, packet_time=0.01)

# %% Generating plots
generate_xput_plots(montecarlo=10, num_n_end=20,
                    num_p=1000, duration=100, num_n_delta=2)
