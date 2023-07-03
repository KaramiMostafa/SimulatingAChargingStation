# **Management and Content Delivery for Smart Networks**
## **Algorithms and Modeling**
This project simulates the operation of a renewable powered charging station for electric vehicles (EVs) in a goods delivery service scenario. The charging station serves a fleet of electric vans and can draw energy from the electric power grid as well as from a set of photovoltaic (PV) panels.

## **Project Structure**
The project consists of four Python files, each related to a specific task within the project. Here's an overview of the files:

- ***task1.py***: Simulates a simple case where the charging station has a small size (NBSS) and assumes a fixed average arrival rate for the EVs. It analyzes the average waiting delay or the missed service probability.

- ***task2.py***: Simulates the system operation over a 24-hour period, considering variable arrival rates of EVs depending on the time of the day. It tests different minimum charge levels (Bth) for picking up batteries from the BSS and evaluates the impact on system performance.

- ***task3.py***: Simulates the system operation on a summer day and a winter day. It considers the presence of PV panels and evaluates the system performance under different PV panel sizes (SPV). It also analyzes the effect of combining different PV panel sizes with the charging station size (NBSS).

The simulation results are presented through various performance metrics, which can be visualized using graphs. The README file doesn't specify the exact performance metrics to be evaluated, so it's up to the user to define and plot relevant metrics based on their specific needs.

The analysis of the system performance can be conducted by comparing different simulation settings and parameters, such as the average arrival rate, minimum charge level, PV panel size, and charging station size. The README file also suggests investigating the warm-up transient period and evaluating the confidence level of the estimated performance metrics.