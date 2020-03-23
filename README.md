# DynaMAC
ECE257B Project - ML based MAC selection protocol

# Description
This repo contains the simulation code used for simulating and evaluating DynaMAC - an RL based MAC Selection Engine. 

# Requirements
Python with numpy installed

# Code
The main functions involved in simulating DynaMAC are:

- 'generate_events'
- 'csma_simulator'
- 'tdma_simulator'
- 'qlearning_egreedy'
- 'qlearning_boltzmann'
- 'decision_final'
- 'Dynamac_SOMAC_test'

# Steps to run the code
- Open dynamac_switch_test.py
- This file has parameters like number of nodes, number of packets per node per second, high traffic duration, low_traffic duration and the window size after which MAC decision is re-computed and updated.
- Set the parameters to match the network you wish to simulate for
- Run the entire file
- The current version of the code simulates the throughput and latency of the network:
-- If only TDMA is employed
-- If only CSMA is employed
-- DynaMAC with eplison-greedy approach
-- DynaAMC with Softmax approach
-- Plots for metric comparison across all the network conditions mentioned above.

# Python Notebook

A sample version of the above test and results are also included in a python notebook and inluded in the repository
