#First Task:

# Import libraries
import random
from queue import PriorityQueue
import pandas as pd
from scipy.stats import ttest_ind
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set working directory:
fn = inspect.getframeinfo(inspect.currentframe()).filename
os.chdir(os.path.dirname(os.path.abspath(fn)))


# Definning inputs and key characteristics of the system
C = 40              # Total capacity of each battery - 40 kWh given
MAX_iC = 10        # Max Initial Charge of arriving batteries - assumed
R = 15              # Rate of Charging at the SCS - 10 kW to 20 kW given
M_C = 40            # Minimum Charge needed for EV to be picked up - given
N_BSS = 2           # Number of Charging points at the Station - assumed

INTER_ARR = 1       # Inter Arrival Time of EVs - constant assumed
W_MAX = 1        # Maximum wait time of EVs before missed service - assumed

SIM_TIME = 200      # Number of Hours that will be sequentially simulated


# Set and Reset Initial Simulation Parameters
# Counts of EVs and Customers in different queues
bat_cnt_charging = 0
bat_cnt_in_charge_queue = 0 
bat_cnt_in_standby = 0
EV_cnt_in_queue = 0 

# Details of EVs and Customers in different queues for tracking purpose
charge_queue = []
charging_bats = []
standby_queue = []
EV_queue = []

# Function to reset the initial parameters for different runs of the  simulation
def refresh_initial_params():
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global bat_cnt_in_standby
    global EV_cnt_in_queue
    global charge_queue
    global charging_bats
    global standby_queue
    global EV_queue
    
    bat_cnt_charging = 0
    bat_cnt_in_charge_queue = 0
    bat_cnt_in_standby = 0
    EV_cnt_in_queue = 0
    
    charge_queue = []
    charging_bats = []
    standby_queue = []
    EV_queue = []
    

# Classes to define the various objects that need to be tracked - EVs, Batteries and Measurements
# EVs defined by their arrival time and expected renege time
class Van:
    def __init__(self, EV_id, arrival_time):
        self.id = EV_id
        self.arrival_time = arrival_time
        self.renege_time = self.arrival_time + W_MAX*1.0001

# Batteries defined by their arrival time and initial charge
class Battery:
    def __init__(self, bat_id, arrival_time):
        self.id = bat_id
        self.arrival_time = arrival_time
        self.initial_charge = round(random.uniform(0,MAX_iC), 2)

# Measurement objects for system performance metrics and to maintain list of EVs and Batteries
class Measure:
    def __init__(self):
        
        self.EVs = []
        self.Batteries = []
        
        self.serviced_EVs = []
        
        self.EVarr = 0
        self.EV_serviced = 0
        self.EV_reneged = 0
        
        self.bats_recharged = 0
        
        self.EV_waiting_delay = []
        self.EV_avg_waiting_delay = []
        
        self.EV_missed_service_prob = []
        
        #self.avg_energy_cost_per_EV = []
        
        self.EV_queue_length = []
        self.avg_EV_queue_length = []
        
        self.charge_queue_length = []
        self.avg_charge_queue_length = []
        
        self.standby_queue_length = []
        self.avg_standby_queue_length = []


# Define functions to handle events in the system
# Function to handle arrival event - EVs arrive, if recharged batteries are available they swap and if there is no charge queue, the discharged battery is recharged
def arrival(EV_id, time, FES, charge_queue, EV_queue, standby_queue):
    global bat_cnt_in_standby
    global bat_cnt_in_charge_queue
    global bat_cnt_charging
    global EV_cnt_in_queue
    global data
        
    EV = Van(EV_id, time)
    
    bat_id = 'B' + EV_id[1:]
    bat = Battery(bat_id, time)
    
    #print(EV_id + ' arrived at ' + str(time) + ' with ' + bat_id)
    
    # Update relevant counts and parameters for tracking
    data.EVarr += 1
    EV_cnt_in_queue += 1
    
    data.EVs.append(EV)
    data.Batteries.append(bat)
    
    EV_queue.append(EV_id)
    
    # Update system performance measurement metrics
    data.EV_queue_length.append({'time': time, 'qlength': len(EV_queue)})
    data.avg_EV_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.EV_queue_length]) / len(data.EV_queue_length)})
    
    data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
    
    # Schedule next arrival of EV
    inter_arrival = INTER_ARR
    FES.put((time + inter_arrival, 'E' + str(int(EV_id[1:]) + 1), "EV Arrival"))
    
    # Schedule renege of the EV
    FES.put((EV.renege_time, EV.id, "Renege"))
     
    
    # If there are recharged batteries available when an EV arrives, the exchange happens
    if bat_cnt_in_standby > 0:
        
        # Update relevant counts and parameters for tracking
        delivered_bat = standby_queue.pop(0)
        #print(EV_id + ' exchanges ' + bat_id + ' for ' + delivered_bat.pop(0) + ' and leaves')
        bat_cnt_in_charge_queue += 1
        bat_cnt_in_standby -= 1
        EV_cnt_in_queue -= 1
        
        charge_queue.append(bat_id)
        EV_queue.remove(EV_id)
                
        # Update system performance measurement metrics
        
        data.serviced_EVs.append(EV.id)
        data.EV_serviced += 1
        
        data.EV_waiting_delay.append({'time': time, 'wait_delay': time - EV.arrival_time})
        data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
        
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
        data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
        
        data.EV_queue_length.append({'time': time, 'qlength': len(EV_queue)})
        data.avg_EV_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.EV_queue_length]) / len(data.EV_queue_length)})        
        
        # If the exchange happens and there is a charging point free, then charging of the discharged battery starts
        if bat_cnt_charging < N_BSS:
        
            # Determine recharge time
            initial_charge = bat.initial_charge
            charging_rate = round(R, 2)
            recharge_time = round((C - initial_charge) / charging_rate, 2)
            
            # Update paramters used to track the simulation
            charging_bats.append(bat_id)
            charge_queue.remove(bat_id)
            #print(bat_id + ' is charging at ' + str(round(time,2)))
            bat_cnt_in_charge_queue -= 1
            bat_cnt_charging += 1
            
            # Update performance metric
            data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
            data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
            
            # Schedule completion of charging process
            FES.put((time + recharge_time, bat.id, "Stand By"))
        
        
# Function to handle completion of charging of batteries
def bat_charge_completion(time, bat_id, FES, charge_queue, standby_queue):
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global bat_cnt_in_standby
    global data
    
    # Update relevant counts and parameters for tracking
    standby_queue.append(bat_id)
    charging_bats.remove(bat_id)
    #print(bat_id + ' charge completed at ' + str(round(time,2)))
    bat_cnt_charging -= 1
    bat_cnt_in_standby += 1
    
    # Update performance metrics
    data.bats_recharged += 1
    data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
    data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
    
    # Inititate the charging process of next battery in queue
    if bat_cnt_in_charge_queue > 0:
        
        bat_id = charge_queue.pop(0)
        bat = [x for x in data.Batteries if x['id'] == bat_id][0]
        
        # Determine recharge time
        initial_charge = bat.initial_charge
        charging_rate = round(R, 2)
        recharge_time = round((C - initial_charge) / charging_rate, 2)
        
        # Update paramters used to track the simulation
        charging_bats.append(bat_id)
        #print(bat_id + ' is charging at ' + str(round(time,2)))
        bat_cnt_in_charge_queue -= 1
        bat_cnt_charging += 1
        
        # Update system performance measurement metrics
                
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        # Schedule completion of charging process
        FES.put((time + recharge_time, bat.id, "Stand By"))
        
# Function to handle exchanging of discharged batteries for recharged batteries when thre are EVs in queue and a recharged battery becomes available
def battery_exchange(time, bat_id, FES, EV_queue, standby_queue):
    global EV_cnt_in_queue
    global bat_cnt_in_standby
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global data
    
    # If there are EVs in the queue and a recharged battery becomes available, then exchange event occurs.
    if len(EV_queue) > 0:
        
        bat_id = standby_queue.pop(0)
        
        EV_id = EV_queue.pop(0)
        #print(EV_id + ' takes ' + bat_id + ' and leaves at ' + str(round(time,2)))
        
        EV = [x for x in data.EVs if x.id == EV_id][0]
        
        # Update relevant tracking parameters
        EV_cnt_in_queue -= 1
        bat_cnt_in_standby -= 1
        
        bat_id = 'B' + EV_id[1:]
        bat = [x for x in data.Batteries if x.id == bat_id][0]
        
        bat_cnt_in_charge_queue += 1
        charge_queue.append(bat_id)
        
        # Update system performance metrics
        
        data.serviced_EVs.append(EV_id)
        data.EV_serviced += 1
        
        data.EV_waiting_delay.append({'time': time, 'wait_delay': time - EV.arrival_time})
        data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
        
        data.EV_queue_length.append({'time': time, 'qlength': len(EV_queue)})
        data.avg_EV_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.EV_queue_length]) / len(data.EV_queue_length)})        
        
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
        data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
        
        # If there is a charging point available, then start charging the received discharged battery                
        if bat_cnt_charging < N_BSS:
            
            # Determine recharge time
            initial_charge = bat.initial_charge
            charging_rate = round(R, 2)
            recharge_time = round((C - initial_charge) / charging_rate, 2)
            
            # Update paramters used to track the simulation
            charging_bats.append(bat_id)
            charge_queue.remove(bat_id)
            #print(bat_id + ' is charging at ' + str(round(time,2)))
            bat_cnt_in_charge_queue -= 1
            bat_cnt_charging += 1
            
            # Update system performance metrics
            data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
            data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
            
            # Schedule completion of charging process
            FES.put((time + recharge_time, bat.id, "Stand By"))
        


# Retrieve Electricity Price data


def get_electricity_prices(filename='electricity_prices.csv'):
    
    prices = pd.read_csv(filename, header=None)
    
    spring_prices = prices.iloc[:,[1,2,3]]
    spring_prices.columns = ['Hour', 'Season', 'Price']
    
    summer_prices = prices.iloc[:,[1,4,5]]
    summer_prices.columns = ['Hour', 'Season', 'Price']
    
    fall_prices = prices.iloc[:,[1,6,7]]
    fall_prices.columns = ['Hour', 'Season', 'Price']
    
    winter_prices = prices.iloc[:,[1,8,9]]
    winter_prices.columns = ['Hour', 'Season', 'Price']

    electricity_prices = spring_prices.append([summer_prices, fall_prices, winter_prices]).reset_index(drop=True)
    electricity_prices['Season'] = electricity_prices['Season'].apply(lambda x: x.replace(":",""))
    
    return electricity_prices

# Run a simulation from time = 0 to SIM_TIME

def main_simulation(RANDOM_SEED):
    
    global bat_cnt_in_charge_queue
    global bat_cnt_in_standby
    global EV_cnt_in_queue
    global data
    
    # Set seed
    random.seed(RANDOM_SEED)
    
    #electricity_prices = get_electricity_prices()
    
    data = Measure()
    
    # Define event queue    
    FES = PriorityQueue()
    
    # Initialize starting events
    time = 0
    FES.put((INTER_ARR, 'E0', "EV Arrival"))
    
    bat_cnt_in_standby = N_BSS
    standby_queue = ['IB' + str(i) for i in range(bat_cnt_in_standby)]
    
    # Starting performance metrics
    data.standby_queue_length.append({'time': time, 'qlength': len(standby_queue)})
    data.avg_standby_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.standby_queue_length]) / len(data.standby_queue_length)})
    
    # Simulate until defined simulation time
    while time < SIM_TIME:
        
        # Get the immediate next scheduled event
        (time, client_id, event_type) = FES.get() 
        
        # If the event is an EV renege event, 
        if event_type == "Renege":
            
            if client_id not in data.serviced_EVs:
                
                # Update relevant simulation tracking parameters
                EV_queue.remove(client_id)
                #print(client_id + ' misses service due to long wait time and leaves.')
                EV_cnt_in_queue -= 1
                
                # Update system performance measurement metrics
                data.EV_reneged += 1
                data.EV_missed_service_prob.append({'time': time, 'prob': round(data.EV_reneged / data.EVarr, 2)})
                
                data.EV_queue_length.append({'time': time, 'qlength': len(EV_queue)})
                data.avg_EV_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.EV_queue_length]) / len(data.EV_queue_length)})        
                
                next
        
        # For other events, call the corresponding event handler function
        elif event_type == "EV Arrival":
            arrival(client_id, time, FES, charge_queue, EV_queue, standby_queue)
        elif event_type == "Stand By":
            bat_charge_completion(time, client_id, FES, charge_queue, standby_queue)
            battery_exchange(time, client_id, FES, EV_queue, standby_queue)



# Defining support functions to help analyze and visualize the simulation results
# Plots a performance metric over time
def plot_over_time(metric, title, ylabel):

    fig = plt.figure()
    ax = plt.axes()
    
    y = [t['avg_wait_delay'] for t in metric]
    x = [t['time'] for t in metric]
    
    ax.plot(x,y)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    
    fig.show()


# Defining functions to identify warm-up transient period in performance metrics

# Fishman's method - graphical method based on column means
def fishmans_method(avg_waiting_delay_list):
    
    metric = []
    time = []
    for i in range(len(avg_waiting_delay_list)):
        metric.append([x['avg_wait_delay'] for x in avg_waiting_delay_list[i]])
        time.append([x['time'] for x in avg_waiting_delay_list[i]])
    
    avg_metric = pd.DataFrame(metric).dropna(axis=1).mean().tolist()
    avg_time = pd.DataFrame(time).dropna(axis=1).mean().tolist()
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(avg_time, avg_metric)
    plt.title('Smoothed based on Column Means - Fishman Approach')
    plt.xlabel('Time')
    plt.ylabel('Average Waiting Delay of EVs (in hours)')
    
    fig.show()
    
    return avg_metric, avg_time

           
# Marginal Standard Error Rule (MSER) - assumes that observations in second half are closer to steady state mean
def mser_method(smoothed_metric, smoothed_time):
    
    # Truncation point is selected such that it minimizes squared error between the selected points and the mean of the selected points
    mser = []
    for d in range(int(len(smoothed_metric) - 1)):
    
        trunc_metric_list = smoothed_metric[(d+1):]
        avg_trunc_metric = sum(trunc_metric_list) / len(trunc_metric_list)
        
        mser.append(sum([(x - avg_trunc_metric)**2 for x in trunc_metric_list]) / (len(trunc_metric_list) ** 2))
    
    
#    fig = plt.figure()
#    ax = plt.axes()
#    ax.plot(mser[:-10])
#    plt.title('MSER Approach for identifying truncation point')
#    plt.xlabel('Index')
#    plt.ylabel('MSER')
#    
#    fig.show()
    
    
    trunc_idx = mser[:-10].index(min(mser[:-10]))
    return smoothed_time[trunc_idx]

# Randomization test - statistical method based on significant differences bewteen sequential batches of values
def randomization_test(smoothed_metric, smoothed_time):
    
    N = int(len(smoothed_metric) / 20)
    avg_metric_batchmeans = [sum(smoothed_metric[x:min(len(smoothed_metric), (x+N))]) / len(smoothed_metric[x:min(len(smoothed_metric), (x+N))]) for x in range(0, len(smoothed_metric), N)]
    for i in range(1,len(avg_metric_batchmeans)):
        batch1 = avg_metric_batchmeans[:i]
        batch2 = avg_metric_batchmeans[i:]
        
        ttest, pval = ttest_ind(batch1, batch2)
        if pval > 0.05:
            break
        
    return smoothed_time[(i-1)*N]


# Functions to calculate confidence interval of a performance metric
# Generic confidence interval function
def mean_CI(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,4), round(m-h,4), round(m+h,4)

# Truncating transient period and calculating confidence interval of the mean of the performance metric
def confidence_interval(metric_list, transient_cutoff_time = 50):
    
    metric = []
    time = []
    for i in range(len(metric_list)):
        metric.append([x['avg_wait_delay'] for x in metric_list[i]])
        time.append([x['time'] for x in metric_list[i]])
    
    metric_means = []
    for i in range(len(metric)):        
        cutoff_idx = next(x for x, val in enumerate(time[i]) if val > transient_cutoff_time)
        metric_means.append(sum(metric[i][cutoff_idx:]) / len(metric[i][cutoff_idx:]))
        
    return mean_CI(metric_means)


#Single Simulation Run
refresh_initial_params()
RANDOM_SEED = 42
data = Measure()
main_simulation(RANDOM_SEED)
avg_waiting_delay = data.EV_avg_waiting_delay
plot_over_time(metric = avg_waiting_delay, 
               title = 'Average Waiting Delay of EVs from Simulation time = 0', ylabel = 'Avg. Waiting Delay')

# Get point estimate of the performance metric
transient_cutoff_time = 50
cutoff_idx = next(x for x, val in enumerate([x['time'] for x in avg_waiting_delay]) if val > transient_cutoff_time)
point_estimate_mean = sum([x['avg_wait_delay'] for x in avg_waiting_delay][cutoff_idx:]) / len(avg_waiting_delay[cutoff_idx:])


# Multiple Independent Runs with different random seeds and different simulation runs
runs = [100, 200, 300, 500, 750, 1000]

print('\nConfidence Intervals of Average Wait Delay')
print('Format: (Expected Value, Lower CI, Upper CI)\n')
smoothed_metric_over_time = []
for N_runs in runs:
    
    # Multiple Independent Runs
    avg_waiting_delay_list = []
    for i in range(N_runs):
        RANDOM_SEED = int(random.random()*10000)
        
        refresh_initial_params()
        data = Measure()
        main_simulation(RANDOM_SEED)
        avg_waiting_delay_list.append(data.EV_avg_waiting_delay)
    
    smoothed_metric, smoothed_time = fishmans_method(avg_waiting_delay_list)
    mser_method(smoothed_metric, smoothed_time)
    randomization_test(smoothed_metric, smoothed_time)
    
    smoothed_metric_over_time.append([smoothed_metric, smoothed_time])
    
    print('Number of Simulation Runs = ' + str(N_runs) + ' and simulation time = ' + str(SIM_TIME))
    print(confidence_interval(avg_waiting_delay_list))
    print('CI Width: ' + str(confidence_interval(avg_waiting_delay_list)[2] - confidence_interval(avg_waiting_delay_list)[1]))
    print('')