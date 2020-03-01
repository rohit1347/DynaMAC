# %% adding imports
from mac_simulator import *

# %% Generating plots
generate_xput_plots(montecarlo=2, num_n_end=10, num_p=2, duration=1e6)


# %%
CSMA_simulator(duration=1e6, packet_time=0.01)


# %%
