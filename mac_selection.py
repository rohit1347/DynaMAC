# %%
# Add imports
from mac_simulator import *
from tdma_simulator import *
from qlearning_optimized import *
# %%


def DynaMAC_switch_test(num_p=5, num_n=5, window_size=10, round=2, duration=200, reg=0):
    latency_array = np.zeros(1)
    xput_array = np.zeros(1)
    MAC_flag = 0
    latency_tracker = None
    pre_packets = None
    if reg:
        simEvents = generate_events_reg(
            num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=round)
    else:
        simEvents = generate_events(
            num_nodes=num_n, num_packets=num_p, sim_end_time=duration, round=round)
    tp = num_p*num_n
    simEvents_length = simEvents.shape[1]
    for iteration, window_start in enumerate(range(0, duration, window_size)):
        # simEvents_plot(simEvents, iteration+1, duration=duration, flag=True)
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


def DynaMAC_somac_test(num_p=5, num_n=5, window_size=10, round=2, duration=200, simEvents=None, mac_init=0, somac_en=1,opt_type = 2,mode=2):
    latency_array = np.zeros(1)
    xput_array = np.zeros(1)
    MAC_array = np.zeros(1)
    MAC_flag = mac_init  # 0 - CSMA , 1 - TDMA
    latency_tracker = None
    pre_packets = None
    g_dt = -2
    simEvents_pre = simEvents
    simEvents_length = simEvents.shape[1]
    for iteration, window_start in enumerate(range(0, duration, window_size)):
        # print(simEvents)
        if(iteration == 0):
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
                slot_time=0.05)
        latency_array = np.append(latency_array, latency)
        xput_array = np.append(xput_array, min(xput,100))
        if(opt_type==0):
            decision_class = decision_final(
                xput_array, 1, mode=MAC_flag, g_dt=g_dt)
        elif(opt_type==1):
            decision_class = decision_final(
                latency_array, 0, mode=MAC_flag, g_dt=g_dt)
        else:
            decision_class = decision_final(
                (xput_array/100.0 - latency_array/5.0), 1, mode=mode, g_dt=g_dt)            
#        decision_class = decision_final(
#            xput_array, 1, mode=MAC_flag, g_dt=g_dt)
#        decision_class = decision_final(
#            latency_array, 0, mode=MAC_flag, g_dt=g_dt)
#         MAC_flag, g_dt = decision_class.result_calc()
        if(somac_en == 1):
            MAC_flag, g_dt = decision_class.decision, decision_class.g_dt
        print(MAC_flag)
        MAC_array = np.append(MAC_array, MAC_flag)

    return latency_array, xput_array, MAC_array, simEvents_pre


def generate_xput_plots(num_p=5, num_n_start=5, num_n_delta=5, num_n_end=50, montecarlo=1, duration=1):
    assert montecarlo >= 1, 'Number of montecarlo runs must be atleast 1'
    num_ns = range(num_n_start, num_n_end, num_n_delta)
    xputs = np.zeros(shape=(montecarlo, len(num_ns)))
    total_packets = np.zeros(shape=(montecarlo, len(num_ns)))
    mc_latency = np.zeros(shape=(montecarlo, len(num_ns)))
    for it in range(0, montecarlo):
        for count, num_n in enumerate(num_ns):
            latency_window_array, xput_window_array = DynaMAC_switch_test(
                num_p=num_p, num_n=num_n, duration=100, reg=1)
            latency, xput = np.mean(
                latency_window_array), np.mean(xput_window_array)
            # latency = np.sum(xput_window_array)
            # tp, xput, latency, _ = CSMA_simulator(
            #     num_p=num_p, num_n=num_n, duration=duration)
            # total_packets[it, count] = tp
            xputs[it, count] = xput
            mc_latency[it, count] = latency
    xputs_mean = np.mean(xputs, axis=0)
    # total_packets_mean = np.mean(total_packets, axis=0)
    xputs_var = np.var(xputs, axis=0)
    latency_mean = np.mean(mc_latency, axis=0)
    latency_var = np.var(mc_latency, axis=0)
    fig1 = plt.figure(num=1, figsize=[9, 6])
    plt.errorbar(num_ns, xputs_mean,
                 yerr=xputs_var, label='Throughput')
    plt.fill_between(num_ns, xputs_mean-xputs_var,
                     xputs_mean+xputs_var, alpha=0.5)
    plt.grid(True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Normalized Throughput')
    plt.legend(loc='lower right')
    plt.title('CSMA: Throughput vs Number of Nodes')
    fig1.show()
    print(latency_mean, latency_var)
    fig2 = plt.figure(num=2, figsize=[9, 6])
    plt.errorbar(num_ns, latency_mean,
                 yerr=latency_var, label='Latency', color='red')
    plt.fill_between(num_ns, latency_mean-latency_var,
                     latency_mean+latency_var, alpha=0.5, facecolor='r')
    plt.grid(True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Latency/packet (s)')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right')
    plt.title('CSMA: Average Latency vs Number of Nodes')
    fig2.show()
