# %% adding imports
from mac_simulator import *
from tdma_simulator import *
from mac_selection import *
from qlearning_optimized import *
import matplotlib.pyplot as plt
# %%
latency_final, xput_final, prot_arr, simEvents = DynaMAC_somac_test(
    num_n=30, num_p=100, duration=300)

# %%
plt.figure()
plt.plot(prot_arr)
plt.show()
plt.figure()
plt.plot(xput_final)
plt.show()
# %%
latency_final, xput_final = DynaMAC_switch_test(
    num_n=30, num_p=40, duration=200)
# %%
#_, xput, _, lat = CSMA_simulator(duration=500, packet_time=0.01)

# %% Generating plots
#generate_xput_plots(montecarlo=2, num_n_end=10, num_p=2, duration=1e6)
