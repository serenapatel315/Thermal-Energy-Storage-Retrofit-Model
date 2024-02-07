# IMPORT RELEVANT PACKAGES

import numpy as np
import pandas as pd
import os
import seaborn as sns
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager



# create your custom colormap 
global technology_color
technology_color = {'Solar':'tab:orange','Wind':'tab:cyan','Coal':'tab:brown'
                    ,'Li-ion':'tab:purple','TES':'tab:green','Retire':'tab:red'}

# generate colors 
def gen_colors(df):
    return [technology_color[col] for col in df.columns]

N=23
T=168

### Plot formatting
def plotparams():
    """
    Format plots
    """
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = "15"
    plt.rcParams['xtick.major.pad']='8'
    plt.rcParams['ytick.major.pad']='8'

    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 'x-large'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 'large'
    plt.rcParams['ytick.labelsize'] = 'large'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['savefig.dpi'] = 900
    plt.rcParams['savefig.bbox'] = 'tight'
    # plt.rcParams['figure.figsize'] = 6.4, 4.8 # 1.33, matplotlib default
    # plt.rcParams['figure.figsize'] = 4.792, 3.458 # 1.38577, Igor my default
    # plt.rcParams['figure.figsize'] = 5.0, 4.0 # 1.25
    plt.rcParams['figure.figsize'] = 5.0, 3.75 # 1.33, fits 4x in ppt slide
    plt.rcParams['xtick.major.size'] = 4 # default 3.5
    plt.rcParams['ytick.major.size'] = 4 # default 3.5
    plt.rcParams['xtick.minor.size'] = 2.5 # default 2
    plt.rcParams['ytick.minor.size'] = 2.5 # default 2
    global font
    font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=15)





# function to make the min hours sensitivity dict from a given directory and the index number


def create_sensitivity_dict(dir):
    results_dir = dir+'results/'
    #results files
    files = [item.replace('.csv','') for item in os.listdir(results_dir)]

    #initialize scenario dictionary 
    scenarios_dict = {}

    T=168

    #for each file in the results folder
    for file in files:

    #check if it's a results file
        if 'results' in file:
            #read the file and set to Dataframe 
            locals()[file] = pd.read_csv(results_dir+file + '.csv').T.reset_index(drop=True)
            #rename to "results"
            results = locals()[file]
            #store filename here 
            filename_n = file + '.csv'
            #print results
        
            #initialize a dictionary to put results in  
            new_dict = dict()
        
            #iterate over the columns of results df
            for j in np.arange(len(results.T)):
                #print(j)
                #if this is a filename
                if 'powerTurbine_t' in results.iloc[0,j] :
                    new_dict[results.iloc[0,j]] = 0
                    #null column
                    results[j][1] = 0
                    #print(j)
                elif pd.isnull(results[j][1]) :
                    new_dict[results.iloc[0,j]] = 0
                    #null column
                    results[j][1] = 0
                    #print(j)
                elif '.csv' in results[j][1]:
                    #set in new dictionary
                    new_dict[results.iloc[0,j]] = results[j][1]
                    
                    #null column
                    results[j][1] = 0

                elif 'LCOE_dict' in results.iloc[0,j]:
                    new_dict[results.iloc[0,j]] = 0
                    #null column
                    results[j][1] = 0
                    #print(j)
                
                else:
                    #populate the dictionary 
                    new_dict[results.iloc[0,j]] = json.loads(results[j][1].replace(";;","").replace(";",",").replace("Any",""))
                    #reshape for these rep weeks
                if type(new_dict[results.iloc[0,j]]) == list: 

                    #calculate N (number of rep weeks)
                    N = int(len(new_dict[results.iloc[0,j]])/T)
                    
                    if len(new_dict[results.iloc[0,j]])==T*N :
                        #print('N:', N)
                        #print(results.iloc[0,j])
                        new_dict[results.iloc[0,j]] = pd.DataFrame(np.array(new_dict[results.iloc[0,j]]).reshape(T,N))
                
        
            #pull values from the dictionary 
            scenario = new_dict['fname'][50:].replace('.csv','') #file 

            #dict of dicts
            scenarios_dict[scenario] = new_dict

            
    return scenarios_dict
            

def create_UP_min_hours_dict(scenarios_dict,dir,dir_index):
     
   profitable_scenario_keys = [key for key in scenarios_dict.keys() if scenarios_dict[key]['objective_value'] < 0]

   sensitivity_keys = [round(float(pd.read_csv(dir+scenarios_dict[key]['fname'][dir_index:])['min_hours'][0]),1) for key in scenarios_dict.keys()]
     
     #initialize dict
   min_hours_dict = {}
     #set the keys to the unique values in the sensitivity keys
     
   min_hours_keys = pd.Series(sensitivity_keys).unique()

        #for each key in the tax credit keys, make a dictionary 
   for sense_key in min_hours_keys:
      min_hours_dict[sense_key] = {}

                # for each key in the scenarios dictionary key, where each key is the same as the parameters input file
      for scen_key in scenarios_dict.keys(): 
            #read the parameters file associated with the scenario
         params = pd.read_csv(dir+scenarios_dict[scen_key]['fname'][dir_index:])

            #get the groupID if Uttar Pradesh
         if 'UttarPradesh' in dir: 
            GroupID = params['Group_ID'][0]

            #add the groupID to scenario dictionary 
            scenarios_dict[scen_key]['Group ID']=GroupID
            
            #if the tax credit is the same as the tax credit key, then re-order in the dicitonary sorted by tax credit sensitivity 
         if round(float(params['min_hours'][0]),1) == sense_key:
            min_hours_dict[sense_key][scen_key]=scenarios_dict[scen_key]

   return min_hours_dict

#choose repweek
def plot_dispatch_separate(n,results_df):
    plotparams()

    fig = plt.figure(figsize = (10,12))
    gs = fig.add_gridspec(5,1, hspace=0)
    (ax1,ax2,ax3,ax4,ax5) = gs.subplots( sharex=True ) 

    ax1.set_title('Representative Week '+str(n))

    # #demand
    ax1.plot(results_df['demand_t'][0][n],label='Demand',color='tab:purple')
    ax1.set_ylabel('Demand (MW)')#,fontsize=fs) 
    ax1.tick_params(axis='both')#, labelsize = fs)

    def abs_max(x):
        y = abs(max(x))
        return y

    #Charge
    Li_ion_discharge = results_df['a_dis_t'][0][n]/abs_max(results_df['a_dis_t'][0][n])
    ax2.plot(Li_ion_discharge,label='Li-ion discharge',color=technology_color['Li-ion'],linestyle='--')

    TES_discharge = results_df['net_discharge_t'][0][n]/abs_max(results_df['net_discharge_t'][0][n])
    ax3.plot(TES_discharge,label='TES discharge',color=technology_color['TES'],linestyle='--')

    #Discharge
    Li_ion_charge = results_df['a_ch_t'][0][n]/abs_max(results_df['a_ch_t'][0][n])
    ax2.plot(Li_ion_charge,label='Li-ion charge',color=technology_color['Li-ion'])

    TES_charge = results_df['net_charge_t'][0][n]/abs_max(results_df['net_charge_t'][0][n])
    ax3.plot(TES_charge,label='TES charge',color=technology_color['TES'],linestyle='--')


    #SOC
    Li_ion_SOC = results_df['soc_t'][0][n]/abs_max(results_df['soc_t'][0][n])
    ax2.plot(Li_ion_SOC,label='Li-ion SOC',color=technology_color['Li-ion'],linestyle=':')

    ax2.legend(loc='upper right')

    TES_SOC = results_df['tank_energy_t'][0][n]/abs_max(results_df['tank_energy_t'][0][n])
    ax3.plot(TES_SOC,label='TES SOC',color=technology_color['TES'],linestyle=':')

    ax3.legend(prop=font)

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles=handles,loc='upper right', bbox_to_anchor=(1.5, 1),prop=font)

    ax2.set_ylabel('Li-ion Dispatch')#,fontsize=fs)
    ax3.set_ylabel('TES Dispatch')#,fontsize=fs)

    #SOC plot for TES and Li-ion
    ax4.plot(Li_ion_SOC,label='Li-ion SOC',color=technology_color['Li-ion'],linestyle='--')
    ax4.plot(TES_SOC,label='TES SOC',color=technology_color['TES'],linestyle='--')

    ax4.legend(loc='upper right',bbox_to_anchor=(1.25,1))

    #generation

    plt.stackplot(np.arange(T),results_df['p_solar_t'][0][n], 
                results_df['p_wind_t'][0][n], 
                results_df['p_coal_t'][0][n],
                labels=['Solar','Wind','Coal'],colors = [technology_color['Solar'],
                                                         technology_color['Wind'],
                                                         technology_color['Coal']])

    plt.legend(loc='upper right')

    ax4.set_xlabel('Hours in Day')#,fontsize=fs)



def plot_dispatch(n,results_df):
  plotparams()
  fig = plt.figure(figsize = (12,6))
  plt.stackplot(np.arange(T),results_df['p_solar_t'][0][n], 
                results_df['p_wind_t'][0][n], 
                results_df['p_coal_t'][0][n],
                results_df['a_dis_t'][0][n],
                results_df['net_discharge_t'][0][n],
                labels=['Solar','Wind','Coal','Li-ion Discharge','TES Discharge'],
                colors = [technology_color['Solar'],
                          technology_color['Wind'],
                          technology_color['Coal'],
                          technology_color['Li-ion'],
                          technology_color['TES']],
                

                )
  plt.plot(results_df['demand_t'][0][n],label='Demand (MW)',color='black')
  
  plt.stackplot(np.arange(T),
                results_df['a_ch_t'][0][n]*-1,
                results_df['net_charge_t'][0][n]*-1,
                labels = ['Li-ion Charge','TES Charge'],
                colors = [technology_color['Li-ion'],
                          technology_color['TES']],
                alpha = [0.5,0.5])
                #hatch = ['+','+'])
                  # labels = ['Li-ion Charge'])
  plt.legend(prop=font)

  plt.ylabel('Units')#,fontsize=fs)
  plt.xlabel('Hours in Day')#,fontsize=fs)
  plt.title('Representative Week '+str(n))

# plot the storage soc map over the year 

def plot_storage_heatmap(results_df):

    # call on fn_map correctly  results['fn_map'][0]
    # cleaned_string = .replace('[', '').replace(']', '').replace(';;','')
    # fn_map = np.asarray([float(value) for value in cleaned_string.split(';')])
    fn_map = results_df['fn_map'][0]

    #get the params file correctly and set W,T,N,alpha_hot and other parameters as needed

    #for 1 to 52 (each of the weeks)
    SOC_array = []

    alpha_hot = 0.00041666666 #change this
    cp = 1.5
    deltaT_salt = 277

    #for each week in the year 
    for week in np.arange(52):
        ## TO DO: simulation
        #n is the representative week that it maps to 
        n = int(fn_map[week])

        #get charging and discharging, cp, Thot-Tcold, and alpha (self discharge) from the associated rep week: 
        flowSalt_charge_n_t = results_df['flowSalt_charge_t'][0][n]
        flowSalt_discharge_n_t = results_df['flowSalt_discharge_t'][0][n]
                

        #define SOC week
        SOC_week = np.zeros(T)

        #set the first value
        # cleaned_string = results['Q'][0].replace('[', '').replace(']', '').replace(';;','')
        # Q = np.asarray([float(value) for value in cleaned_string.split(',')])
        Q = results_df['Q'][0]
        Q_w = Q[week] 

        SOC_week[0] = Q_w + (flowSalt_charge_n_t[0] - flowSalt_discharge_n_t[0])*cp*deltaT_salt
                #SOC_week[0] = Q_w + new_dict['tank_energy_t'][n][0]
                
        #set the rest of values: 
        for i in np.arange(1,T): 
            SOC_week[i] = (flowSalt_charge_n_t[i] - flowSalt_discharge_n_t[i])*cp*deltaT_salt + (1-alpha_hot)*SOC_week[i-1]
                


        SOC_array = SOC_array + list(SOC_week / results_df["tank_energy_max"][0])


    plotparams()
    plt.figure(figsize = (10,2))
    im=sns.heatmap(pd.DataFrame(np.array(SOC_array).reshape((364,24))).T,yticklabels=12,xticklabels=60,cmap='viridis',vmin=0,vmax=1)


    plt.xlabel('Days')
    plt.ylabel('Hours')



def create_curtailment_df(CEM_min_hours_dict):
    # solar_curtailment_list = []
    # wind_curtailment_list = []
    label_list = []
    curtailment_list = []

    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]

    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()
        # results_df = results.sort_values(by='emissions',ascending=True)
        #calc curtailment
        curtailment = 1- ((results_df["dispatch_Solar"][0]+results_df["dispatch_Wind"][0]) / (results_df["solar_max"][0] + results_df["wind_max"][0]) )
        #curtailment for each scenario
        curtailment_list = np.append(curtailment_list,curtailment)
        # solar_curtailment_list = np.append(solar_curtailment_list, 1- (results_df["dispatch_Solar"][0] / results_df["solar_max"][0]))
        # wind_curtailment_list = np.append(wind_curtailment_list,1- (results_df["dispatch_Wind"][0] / results_df["wind_max"][0]))
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be
    curtailment_df = pd.DataFrame({'Label':label_list,
                                #    'Solar':solar_curtailment_list, 
                                #    'Wind':wind_curtailment_list,
                                   'Curtailment': curtailment_list

                                   }) 

    return curtailment_df

def create_storage_dispatch_df(CEM_min_hours_dict):
    TES_list = []
    Li_list = []
    label_list = []

    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]

    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()

        #curtailment for each scenario
        TES_list = np.append(TES_list, results_df["dispatch_TES"][0] )
        Li_list = np.append(Li_list,results_df["dispatch_Li"][0] )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    storage_dispatch = pd.DataFrame({'Label':label_list,
                                   'TES':TES_list, 
                                   'Li-ion':Li_list,
                                   }) 
    

    return storage_dispatch


def create_P_storage_df(CEM_min_hours_dict):
    TES_list = []
    Li_list = []
    label_list = []

    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]

    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()

        #curtailment for each scenario
        TES_list = np.append(TES_list, results_df["powerTurbine"][0] )
        Li_list = np.append(Li_list,results_df["capP_Li"][0] )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    storage_P = pd.DataFrame({'Label':label_list,
                                   'TES':TES_list, 
                                   'Li-ion':Li_list,
                                   }) 
    

    return storage_P


def create_E_storage_df(CEM_min_hours_dict):
    TES_list = []
    Li_list = []
    label_list = []

    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()

        #curtailment for each scenario
        TES_list = np.append(TES_list, results_df["tank_energy_max"][0] )
        Li_list = np.append(Li_list,results_df["capE_Li"][0] )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    storage_E = pd.DataFrame({'Label':label_list,
                                   'TES':TES_list, 
                                   'Li-ion':Li_list,
                                   }) 
    

    return storage_E


def create_conversions_df(CEM_min_hours_dict):
    TES_list = []
    coal_list = []
    retire_list = []

    label_list = []

    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()

        #curtailment for each scenario
        TES_list = np.append(TES_list, results_df["powerPeak"][0] )
        coal_list = np.append(coal_list,results_df["capCoal"][0] )
        retire_list = np.append(retire_list,results_df["capRetire"][0] )

        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    conversions = pd.DataFrame({'Label':label_list,
                                   'Coal':coal_list, 
                                   'Retire':retire_list,
                                   'TES':TES_list

                                   }) 
    

    return conversions


def create_capacity_df(CEM_min_hours_dict):
    solar_list = []
    wind_list = []
    coal_list = []
    TES_list = []
    Li_list = [] 
    label_list = []

    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()

        #curtailment for each scenario
        solar_list = np.append(solar_list, results_df["capSolar"][0])
        wind_list = np.append(wind_list, results_df["capWind"][0])
        coal_list = np.append(coal_list, results_df["capCoal"][0])
        TES_list = np.append(TES_list, results_df["powerTurbine"][0] )
        Li_list = np.append(Li_list,results_df["capP_Li"][0] )

        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    capacity_df = pd.DataFrame({'Label':label_list,
                                   'Solar':solar_list, 
                                   'Wind':wind_list,
                                   'Coal':coal_list,
                                   'TES': TES_list,
                                   'Li-ion': Li_list
                                   }) 
    

    return capacity_df


def create_generation_df(CEM_min_hours_dict):
    solar_list = []
    wind_list = []
    coal_list = [] 
    label_list = []
    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()

        #curtailment for each scenario
        solar_list = np.append(solar_list, results_df["dispatch_Solar"][0])
        wind_list = np.append(wind_list, results_df["dispatch_Wind"][0])
        coal_list = np.append(coal_list, results_df["dispatch_Coal"][0])

        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    generation_df = pd.DataFrame({'Label':label_list,
                                   'Solar':solar_list, 
                                   'Wind':wind_list,
                                   'Coal':coal_list
                                   }) 
    

    return generation_df


def create_emissions_df(CEM_min_hours_dict):
    emissions_list = []
    label_list = []
    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()
        emissions_list = np.append(emissions_list,results_df["emissions"][0] )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    emissions_df = pd.DataFrame({'Label':label_list,
                                   'emissions':emissions_list,
                                   }) 
        

    return emissions_df


def create_syscost_df(CEM_min_hours_dict):
    syscost_list = []
    label_list = []
    results_dict = dict(sorted(CEM_min_hours_dict[0].items(), key=lambda item: item[1]['emissions'])) #CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()
        syscost_list = np.append(syscost_list, results_df["objective_value"][0]*1000 )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    syscost_df = pd.DataFrame({'Label':label_list,
                                   'system cost':syscost_list, 
                                   }) 
        

    return syscost_df






def plot_stacked(df,path_download):
    plotparams()
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1, wspace=0)
    (ax) = gs.subplots()
    #df has label and values
    # if 'curtail' in path_download:
    #     df.plot(kind='bar',x = 'Label', stacked = False, color = gen_colors(df.drop(['Label'],axis=1)),figsize=(10,5),ax=ax)
    if ('curtail' in path_download)| ('emissions' in path_download) | ('syscost' in path_download):
        df.plot(kind='bar',x = 'Label', stacked = False,figsize=(10,5),ax=ax)

    else:
        df.plot(kind='bar',x = 'Label', stacked = True, color = gen_colors(df.drop(['Label'],axis=1)),figsize=(10,5),ax=ax)
    
    plt.xlabel('Scenario')
    plt.savefig(path_download)

def create_emissions_syscost_df(CEM_min_hours_dict):
    syscost_list = []
    emissions_list = []
    label_list = []
    results_dict = CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()
        #curtailment for each scenario
        syscost_list = np.append(syscost_list, results_df["objective_value"][0]*1000 )
        emissions_list = np.append(emissions_list,results_df["emissions"][0] )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    emissions_syscost_df = pd.DataFrame({'Label':label_list,
                                   'system cost':syscost_list, 
                                   'emissions':emissions_list,
                                   }) 
        

    return emissions_syscost_df

def plot_emissions_systemcost(emissions_syscost_df,path_download):
    plotparams()
    #could convert to emissions percentage if you have the df 
    plt.scatter(emissions_syscost_df['emissions'],emissions_syscost_df['system cost'])
    plt.xlabel('Emissions (tCO2/year)')
    plt.ylabel('System Cost (USD/year)')
    
    plt.savefig(path_download)

    

    
def create_LCOE_df(CEM_min_hours_dict):
    #for each scenario, total LCOE values for each of these 
    TES_LCOS = []
    Li_LCOS = []
    coal_LCOE = []
    solar_LCOE = []
    wind_LCOE = []


    label_list = []

    results_dict = CEM_min_hours_dict[0]

    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()
        #curtailment for each scenario
        TES_LCOS = np.append(TES_LCOS, results_df["TES_LCOE_total"][0] )
        if results_df["dispatch_Li"][0] > 0:
            Li_LCOS = np.append(Li_LCOS, (results_df["CAPEX_Li"][0]*results_df["CRF"][0]
                            + results_df["OPEX_Li"][0]
                            + results_df["FOM_Li"][0]) / results_df["dispatch_Li"][0])
        else:
            Li_LCOS = np.append(Li_LCOS,0)
        if results_df["dispatch_Coal"][0] > 0:
            coal_LCOE = np.append(coal_LCOE, (
                            + results_df["OPEX_Coal"][0]
                            + results_df["FOM_Coal"][0]) / results_df["dispatch_Coal"][0])
        else: 
            coal_LCOE = np.append(coal_LCOE,0)
        
        if results_df["dispatch_Solar"][0]>0:
            solar_LCOE = np.append(solar_LCOE, (results_df["CAPEX_Li"][0]*results_df["CRF"][0]
                            + results_df["OPEX_Li"][0]
                            + results_df["FOM_Li"][0]) / results_df["dispatch_Solar"][0])
        else: 
            solar_LCOE = np.append(solar_LCOE,0)
        
        if results_df["dispatch_Wind"][0]>0:
            wind_LCOE = np.append(wind_LCOE, (results_df["CAPEX_Wind"][0]*results_df["CRF"][0]
                            + results_df["OPEX_Wind"][0]
                            + results_df["FOM_Wind"][0]) / results_df["dispatch_Wind"][0])
        else: 
            wind_LCOE = np.append(wind_LCOE,0)


        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    all_LCOE_df = pd.DataFrame({'Label':label_list,
                                   'TES':TES_LCOS, 
                                   'Li-ion':Li_LCOS,
                                   'Solar':solar_LCOE,
                                   'Coal':coal_LCOE,
                                   'Wind':wind_LCOE,
                                   }) 
        

    return all_LCOE_df


# check relative_gap 
def relative_gap(CEM_min_hours_dict):
    relative_gap_list = []
    emissions_list = []
    label_list = []
    results_dict = CEM_min_hours_dict[0]


    for key in results_dict.keys():
        #results_df = CEM_min_hours_dict
        #create dataframe for stacked plots
        results_df = pd.DataFrame.from_dict(results_dict[key], orient='index').transpose()
        #curtailment for each scenario
        relative_gap_list = np.append(relative_gap_list, results_df["relative_gap"][0] )
        #emissions_list = np.append(emissions_list,results_df["emissions"][0] )
        label_list = np.append(label_list,str(key )) #reset this label to whatever the label needs to be

    relative_gap_df = pd.DataFrame({'Label':label_list,
                                   'relative_gap':relative_gap_list, 
                                   
                                   }) 
        

    return relative_gap_df


def filter_df(df,columns_download,path_download):
    df_filtered = pd.DataFrame(columns=df.columns)
    for label_i in columns_download:
        df_filtered=pd.concat((df[df["Label"] == label_i],df_filtered))
    
    df_filtered.reset_index(drop=True,inplace=True)

    #save
    df_filtered.to_csv(path_download+'.csv',index=False)

    return df_filtered