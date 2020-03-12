# %% adding imports
from mac_simulator import *
from tdma_simulator import *
from mac_selection import *


# %%
print(DynaMAC_switch_test(num_n=30, num_p=100))

# %%
_, xput, _, lat = CSMA_simulator(duration=500, packet_time=0.01)

# %% Generating plots
generate_xput_plots(montecarlo=2, num_n_end=10, num_p=2, duration=1e6)
