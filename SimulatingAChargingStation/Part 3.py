#Tird Task:

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

#Set working directory:
fn = inspect.getframeinfo(inspect.currentframe()).filename
os.chdir(os.path.dirname(os.path.abspath(fn)))


#Defining inputs and the system:
C = 40              # Total capacity of each battery - 40 kWh given
MAX_iC = 10        # Max Initial Charge of arriving batteries - assumed
R = 15              # Rate of Charging at the SCS - 10 kW to 20 kW given
B_TH = 20            # Minimum Charge needed for EV to be picked up during high demand periods
N_BSS = 2           # Number of Charging points at the Station - assumed
W_MAX = 1        # Maximum wait time of EVs before missed service - assumed
window = [19, 20, 21, 22, 23] # Postponing of charging for a fraction of batteries during this period

#Assumptions of variable arrival rates of EVs depending on time of day

EV_arr_rates = [0.25,0.75,1,2] 

INTER_ARR = [EV_arr_rates[0]]*7 + [EV_arr_rates[1]]*6 + [EV_arr_rates[2]]*7 + [EV_arr_rates[3]]*4

partial_charge_times = [13, 14, 15, 16, 17, 18, 19, 20]


#Set and Reset Initial Simulation Parameters
SIM_TIME = 500  # Number of Hours that will be sequentially simulated

# Counts of EVs and Customers in different queues:
bat_cnt_charging = 0
bat_cnt_in_charge_queue = 0 
bat_cnt_in_standby = 0
bat_cnt_postponed = 0
EV_cnt_in_queue = 0 

# Details of EVs and Customers in different queues for tracking purpose:
charge_queue = []
charging_bats = []
standby_queue = []
EV_queue = []

# Function to reset the initial parameters for different runs of the  simulation:
def refresh_initial_params():
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global bat_cnt_in_standby
    global bat_cnt_postponed
    global EV_cnt_in_queue
    global charge_queue
    global charging_bats
    global standby_queue
    global EV_queue
    
    bat_cnt_charging = 0
    bat_cnt_in_charge_queue = 0
    bat_cnt_in_standby = 0
    bat_cnt_postponed = 0
    EV_cnt_in_queue = 0
    
    charge_queue = []
    charging_bats = []
    standby_queue = []
    EV_queue = []
    

#Classes to define the various objects that need to be tracked - EVs, Batteries and Measurements

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
        self.recharge_start_time = 0
        self.recharge_end_time = 0

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
        
        self.recharge_start_times = {}
        self.recharge_end_times = {}
        
        self.util_per_hour = [0]*24
        
        self.EV_queue_length = []
        self.avg_EV_queue_length = []
        
        self.charge_queue_length = []
        self.avg_charge_queue_length = []
        
        self.standby_queue_length = []
        self.avg_standby_queue_length = []

#Defining functions to handle events in the system:

# Function to handle arrival event - EVs arrive, if recharged batteries are available they swap and if there is no charge queue, the discharged battery is recharged
def arrival(EV_id, time, FES, charge_queue, EV_queue, standby_queue):
    global bat_cnt_in_standby
    global bat_cnt_in_charge_queue
    global bat_cnt_charging
    global bat_cnt_postponed
    global EV_cnt_in_queue
    global data
        
    EV = Van(EV_id, time)
    
    bat_id = 'B' + EV_id[1:]
    bat = Battery(bat_id, time)
    
    #print(EV_id + ' arrived at ' + str(round(time,2)) + ' with ' + bat_id)
    
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
    inter_arrival = random.expovariate(INTER_ARR[int(time % 24)])
    FES.put((time + inter_arrival, 'E' + str(int(EV_id[1:]) + 1), "EV Arrival"))
    
    # Schedule renege of the EV
    FES.put((EV.renege_time, EV.id, "Renege"))
     
    
    # If there are recharged batteries available when an EV arrives, the exchange happens
    if bat_cnt_in_standby > 0:
        
        # Update relevant counts and parameters for tracking
        delivered_bat = standby_queue.pop(0)
        #print(EV_id + ' exchanges ' + bat_id + ' for ' + delivered_bat + ' and leaves')
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
            
            if random.choices([1,0], [f, 1-f])[0] == 1 and (int(time) % 24 in window):
                FES.put((time + random.uniform(0,T_max), bat.id, 'Postpone'))
                #print('Charging of ' + bat.id + ' postponed')
                bat_cnt_in_charge_queue -= 1
                bat_cnt_postponed += 1
                
                charge_queue.remove(bat_id)
            
            else:
                # Determine recharge time
                initial_charge = bat.initial_charge
                charging_rate = round(R, 2)
                recharge_time = round((C - initial_charge) / charging_rate, 2)
                availability_alert_time = round((B_TH - initial_charge) / charging_rate, 2)
                
                # Update paramters used to track the simulation
                charging_bats.append(bat_id)
                charge_queue.remove(bat_id)
                #print(bat_id + ' is charging at ' + str(round(time,2)))
                bat_cnt_in_charge_queue -= 1
                bat_cnt_charging += 1
                
                # Update performance metric
                data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
                data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
                
                data.recharge_start_times[bat.id] = time
                
                # Schedule completion of charging process
                FES.put((time + recharge_time, bat.id, "Stand By"))
                FES.put((time + availability_alert_time, bat.id, "Availability Alert"))
        
        
# Function to handle completion of charging of batteries
def bat_charge_completion(time, bat_id, FES, charge_queue, standby_queue):
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global bat_cnt_in_standby
    global bat_cnt_postponed
    global data
    
    if bat_id in charging_bats:
    
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
        
        data.recharge_end_times[bat_id] = time
        
        # Inititate the charging process of next battery in queue
        if bat_cnt_in_charge_queue > 0:
            
            bat_id = charge_queue.pop(0)
            bat = [x for x in data.Batteries if x.id == bat_id][0]
            
            if random.choices([1,0], [f, 1-f])[0] == 1 and (int(time) % 24 in window):
                FES.put((time + random.uniform(0,T_max), bat_id, 'Postpone'))
                #print('Charging of ' + bat_id + ' postponed')
                
                bat_cnt_in_charge_queue -= 1
                bat_cnt_postponed += 1
                
            else:
            
                # Determine recharge time
                initial_charge = bat.initial_charge
                charging_rate = round(R, 2)
                recharge_time = round((C - initial_charge) / charging_rate, 2)
                availability_alert_time = round((B_TH - initial_charge) / charging_rate, 2)
                
                # Update paramters used to track the simulation
                charging_bats.append(bat_id)
                #print(bat_id + ' is charging at ' + str(round(time,2)))
                bat_cnt_in_charge_queue -= 1
                bat_cnt_charging += 1
                
                # Update system performance measurement metrics
                        
                data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
                data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
                
                data.recharge_start_times[bat_id] = time
                
                # Schedule completion of charging process
                FES.put((time + recharge_time, bat.id, "Stand By"))
                FES.put((time + availability_alert_time, bat.id, "Availability Alert"))
        
# Function to handle exchanging of discharged batteries for recharged batteries when thre are EVs in queue and a recharged battery becomes available
def battery_exchange(time, bat_id, FES, EV_queue, standby_queue):
    global EV_cnt_in_queue
    global bat_cnt_in_standby
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global bat_cnt_postponed
    global data
    
    # If there are EVs in the queue and a recharged battery becomes available, then exchange event occurs.
    if bat_id in standby_queue and len(EV_queue) > 0:
        
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
            
            if random.choices([1,0], [f, 1-f])[0] == 1 and (int(time) % 24 in window):
                FES.put((time + random.uniform(0,T_max), bat_id, 'Postpone'))
                #print('Charging of ' + bat_id + 'postponed')
                
                bat_cnt_in_charge_queue -= 1
                bat_cnt_postponed += 1
                
                charge_queue.remove(bat_id)
                
            else:
            
                # Determine recharge time
                initial_charge = bat.initial_charge
                charging_rate = round(R, 2)
                recharge_time = round((C - initial_charge) / charging_rate, 2)
                availability_alert_time = round((B_TH - initial_charge) / charging_rate, 2)
                
                # Update paramters used to track the simulation
                charging_bats.append(bat_id)
                charge_queue.remove(bat_id)
                #print(bat_id + ' is charging at ' + str(round(time,2)))
                bat_cnt_in_charge_queue -= 1
                bat_cnt_charging += 1
                
                # Update system performance metrics
                data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
                data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
                
                data.recharge_start_times[bat_id] = time
                
                # Schedule completion of charging process
                FES.put((time + recharge_time, bat.id, "Stand By"))
                FES.put((time + availability_alert_time, bat.id, "Availability Alert"))

# Function to handle event of swapping partially recharged batteries
def availability_alert(time, bat_id, FES, EV_queue, charge_queue):
    global EV_cnt_in_queue
    global bat_cnt_charging
    global bat_cnt_in_charge_queue
    global bat_cnt_postponed
    global data
    
    # If there are EVs in the queue and a recharged battery becomes available, then exchange event occurs.
    if len(EV_queue) > 0:
        
        EV_id = EV_queue.pop(0)
        #print(EV_id + ' takes ' + bat_id + ' and leaves at ' + str(round(time,2)) + ' - partially charged')
        
        EV = [x for x in data.EVs if x.id == EV_id][0]
        
        # Update performance metric
        data.recharge_end_times[bat_id] = time
        
        # Update relevant tracking parameters
        EV_cnt_in_queue -= 1
        bat_cnt_charging -= 1
        
        data.serviced_EVs.append(EV_id)
        
        charging_bats.remove(bat_id)
        
        bat_id = 'B' + EV_id[1:]
        bat = [x for x in data.Batteries if x.id == bat_id][0]
        
        bat_cnt_in_charge_queue += 1
        charge_queue.append(bat_id)
        
        # Initiate charging of the next battery in queue
        bat_id = charge_queue.pop(0)
        bat = [x for x in data.Batteries if x.id == bat_id][0]
        
        if random.choices([1,0], [f, 1-f])[0] == 1 and (int(time) % 24 in window):
            FES.put((time + random.uniform(0,T_max), bat_id, 'Postpone'))
            #print('Charging of ' + bat_id + 'postponed')
            
            bat_cnt_in_charge_queue -= 1
            bat_cnt_postponed += 1    
                
        else:
        
            # Determine recharge time
            initial_charge = bat.initial_charge
            charging_rate = round(R, 2)
            recharge_time = round((C - initial_charge) / charging_rate, 2)
            availability_alert_time = round((B_TH - initial_charge) / charging_rate, 2)
            
            # Update paramters used to track the simulation
            charging_bats.append(bat_id)
            #print(bat_id + ' is charging at ' + str(round(time,2)))
            bat_cnt_in_charge_queue -= 1
            bat_cnt_charging += 1
            
            # Update system performance metrics
            
            data.serviced_EVs.append(EV_id)
            data.EV_serviced += 1
            
            data.EV_waiting_delay.append({'time': time, 'wait_delay': time - EV.arrival_time})
            data.EV_avg_waiting_delay.append({'time': time, 'avg_wait_delay': sum([x['wait_delay'] for x in data.EV_waiting_delay]) / len(data.EV_waiting_delay)})
            
            data.EV_queue_length.append({'time': time, 'qlength': len(EV_queue)})
            data.avg_EV_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.EV_queue_length]) / len(data.EV_queue_length)})        
            
            data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
            data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
            
            data.recharge_start_times[bat_id] = time
            
            # Schedule completion of charging process
            FES.put((time + recharge_time, bat.id, "Stand By"))
            FES.put((time + availability_alert_time, bat.id, "Availability Alert"))

# Function to handle battery recharge postponement event
def postponement(time, bat_id, FES, charge_queue, charging_bats):
    global bat_cnt_in_charge_queue
    global bat_cnt_postponed
    global bat_cnt_charging
    global data
    
    bat_cnt_in_charge_queue += 1
    charge_queue.append(bat_id)
    bat_cnt_postponed -= 1
    
    if bat_cnt_charging < N_BSS: 
        
        bat = [x for x in data.Batteries if x.id == bat_id][0]
        
        # Determine recharge time
        initial_charge = bat.initial_charge
        charging_rate = round(R, 2)
        recharge_time = round((C - initial_charge) / charging_rate, 2)
        availability_alert_time = round((B_TH - initial_charge) / charging_rate, 2)
        
        # Update paramters used to track the simulation
        charging_bats.append(bat_id)
        charge_queue.remove(bat_id)
        #print(bat_id + ' is charging at ' + str(round(time,2)))
        bat_cnt_in_charge_queue -= 1
        bat_cnt_charging += 1
        
        # Update performance metric
        data.charge_queue_length.append({'time': time, 'qlength': len(charge_queue)})
        data.avg_charge_queue_length.append({'time': time, 'avg_qlength': sum([x['qlength'] for x in data.charge_queue_length]) / len(data.charge_queue_length)})
        
        data.recharge_start_times[bat.id] = time
        
        # Schedule completion of charging process
        FES.put((time + recharge_time, bat.id, "Stand By"))
        FES.put((time + availability_alert_time, bat.id, "Availability Alert"))



#Run a simulation from time = 0 to SIM_TIME

def main_simulation(RANDOM_SEED):
    
    global bat_cnt_in_charge_queue
    global bat_cnt_in_standby
    global EV_cnt_in_queue
    global data
    
    # Set seed
    random.seed(RANDOM_SEED)
    
    data = Measure()
    
    # Define event queue    
    FES = PriorityQueue()
    
    # Initialize starting events
    time = 0
    FES.put((random.expovariate(INTER_ARR[int(time % 24)]), 'E0', "EV Arrival"))
    
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
        elif event_type == "Availability Alert" and int(time%24) in partial_charge_times:
            availability_alert(time, client_id, FES, EV_queue, charge_queue)
        elif event_type == "Postpone":
            postponement(time, client_id, FES, charge_queue, charging_bats)


#Retrieve Electricity Price data and PV Production data

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

def get_PV_capacity(filename='PVproduction_PanelSize1kWp.csv'):
    
    pv_production = pd.read_csv(filename)
    
    # Convert to seasons
    def m_to_s(m):
        if m in [3,4,5]:
            return 'Spring'
        elif m in [6, 7, 8]:
            return 'Summer'
        elif m in [9, 10, 11]:
            return 'Fall'
        elif m in [12, 1, 2]:
            return 'Winter'
    
    pv_production['Season'] = pv_production['Month'].apply(lambda x: m_to_s(x))
    
    production_by_season = pv_production.groupby(['Season', 'Hour']).agg({'Output power (W)': 'mean'})
    production_by_season = production_by_season.reset_index()
    
    return production_by_season


#Define support functions to help analyze and visualize the performance metrics:
def util_bars(util_by_hour): # Plots utilization by hour
    
    avg_metric = pd.DataFrame(util_by_hour).mean().tolist()
    avg_time = [x for x in range(24)]
    
    fig = plt.figure()
    ax = plt.axes()
    ax.bar(avg_time, avg_metric)
    plt.title('Avg. BSS Utilization by Hour')
    plt.xlabel('Hour of day')
    plt.ylabel('Utilization %')
    
    fig.show()
    
# Calculates electricity cost based on station hourly utilization
def get_cost(util_by_hour, season, S_PV = 100, N_BSS = 2):
    
    avg_util = pd.DataFrame(util_by_hour).mean().tolist()
    
    avg_power_consumed = [x*R*N_BSS for x in avg_util]
    
    electricity_prices = get_electricity_prices()
    
    production_by_season = get_PV_capacity()
    production_summer = production_by_season.loc[production_by_season['Season'] == 'Summer', 'Output power (W)'].tolist()
    production_winter = production_by_season.loc[production_by_season['Season'] == 'Winter', 'Output power (W)'].tolist()
    
    production_summer = [x*S_PV/1000 for x in production_summer]
    production_winter = [x*S_PV/1000 for x in production_winter]
    
    if season == 'Summer':
        power_from_grid = [max(x-y,0) for x, y in zip(avg_power_consumed, production_summer)]
        prices_summer = electricity_prices.loc[electricity_prices['Season'] == 'SUMMER','Price'].tolist()
        cost = sum([x*y/1000 for x,y in zip(power_from_grid, prices_summer)])
    
    elif season == 'Winter':
        power_from_grid = [max(x-y,0) for x, y in zip(avg_power_consumed, production_winter)]
        prices_winter = electricity_prices.loc[electricity_prices['Season'] == 'WINTER','Price'].tolist()
        cost = sum([x*y/1000 for x,y in zip(power_from_grid, prices_winter)])
        
    return cost

# Full cost analysis for different values of S_PV and N_BSS
def cost_analysis(S_PV = 100, N_BSS = 2):
    
    global data
    
    # Define all performance metrics to be tracked
    util_by_hour = []
    cost_summer = []
    cost_winter = []
    
    # Define number of runs and run the simulation
    N_runs = 200
    for i in range(N_runs):
        
        RANDOM_SEED = int(random.random()*10000)
        
        refresh_initial_params()
        data = Measure()
        main_simulation(RANDOM_SEED)
        
        recharged_bats = list(data.recharge_start_times.keys())
        temp_util_list = [0]*SIM_TIME
        for bat in recharged_bats:
    
            recharge_start_time = data.recharge_start_times[bat]
            if bat in list(data.recharge_end_times.keys()):
                recharge_time = data.recharge_end_times[bat] - data.recharge_start_times[bat]
            else:
                recharge_time = SIM_TIME - data.recharge_start_times[bat]
            
            temp_time = (recharge_start_time - int(recharge_start_time)) + recharge_time
            temp_time_list = [1 for x in range(int(temp_time))] + [round(temp_time - int(temp_time),2)]
            temp_time_list[0] = round(temp_time_list[0] - (recharge_start_time - int(recharge_start_time)),2)
            for j in range(int(recharge_start_time), min(int(recharge_start_time) + len(temp_time_list), SIM_TIME)):
                temp_util_list[j] = temp_util_list[j] + temp_time_list[j - int(recharge_start_time)]
        
        temp_util_list_24 = []
        for j in range(24):
            temp_util_list_24.append(sum([temp_util_list[x] for x in range(j, len(temp_util_list), 24)]) / len([temp_util_list[x] for x in range(j, len(temp_util_list), 24)]))
        
        util_by_hour.append([x/N_BSS for x in temp_util_list_24])
     
        cost_summer.append(get_cost(util_by_hour, 'Summer', S_PV, N_BSS))
        cost_winter.append(get_cost(util_by_hour, 'Winter', S_PV, N_BSS))
    
    cost_summer = sum(cost_summer)/len(cost_summer)
    cost_winter = sum(cost_winter)/len(cost_winter)
    
    return util_by_hour, cost_summer, cost_winter

# First part of the task - Combinations of S_PV and N_BSS
#S_PV_list = [100, 200, 500, 750, 1000]
#N_BSS_list = [2, 3, 5, 7, 10]
#
#cost_summer_list = []
#cost_winter_list = []
#for pv in S_PV_list:
#    temp_list_summer = []
#    temp_list_winter = []
#    for N_BSS in N_BSS_list:
#        util_by_hour, cost_summer, cost_winter = cost_analysis(pv, N_BSS)
#        temp_list_summer.append(cost_summer)
#        temp_list_winter.append(cost_winter)
#        
#    cost_summer_list.append(temp_list_summer)
#    cost_winter_list.append(temp_list_winter)
#    
##util_by_hour, cost_summer, cost_winter = cost_analysis(100, 2)
#    
#pd.DataFrame(cost_summer_list)
#pd.DataFrame(cost_winter_list)


# Seond part of the question - postponement strategy
f_list = [0, 0.25, 0.5, 0.75]
T_max_list = [4,5,6,7,8]

cost_summer_list = []
cost_winter_list = []
serviced_EVs_cnt = []
EV_missed_service_prob = []
for f in f_list:
    temp_list_summer = []
    temp_list_winter = []
    temp_list_serviced_EVs = []
    temp_list_missed_prob = []
    for T_max in T_max_list:
        util_by_hour, cost_summer, cost_winter = cost_analysis()
        temp_list_summer.append(cost_summer)
        temp_list_winter.append(cost_winter)
        temp_list_serviced_EVs.append(len(data.serviced_EVs))
        temp_list_missed_prob.append(data.EV_missed_service_prob[-1]['prob'])
        
        print('f: ' + str(f) + ', ' + 'T_max: ' + str(T_max))
        util_bars(util_by_hour)
        
    cost_summer_list.append(temp_list_summer)
    cost_winter_list.append(temp_list_winter)
    serviced_EVs_cnt.append(temp_list_serviced_EVs)
    EV_missed_service_prob.append(temp_list_missed_prob)
    
pd.DataFrame(cost_summer_list)
pd.DataFrame(cost_winter_list)
pd.DataFrame(serviced_EVs_cnt)
pd.DataFrame(EV_missed_service_prob)