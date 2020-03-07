# %%
# Add imports
from mac_simulator import *
# %%


def DynaMAC_switch_test(num_p=5, num_n=5, duration=10, round=2):
    simEvents = generate_events(
        num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=round)
    tp = num_p*num_n
    simEvents_length = simEvents.shape[1]
    while simEvents.shape[1] > simEvents_length / 2:
