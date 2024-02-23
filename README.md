# Thermal-Energy-Storage-Retrofit-Model

## Model Description

This model represents molten salt based thermal energy storage retrofits of coal plants, leveraging data on the Indian coal fleet evaluates optimal dispatch and sizing of such retrofits across projected price scenarios in 2030 in India as well as different types and vintages of coal plants in India. 

We formulate the Integrated Dispatch and Sizing (IDS) optimization model to evaluate the cost-optimal design and operation of the TES retrofitted coal power plant under dynamic grid electricity prices. The objective function of the model is the annualized profit of the facility, defined as the annual revenue earned from sale of electricity throughout the year minus annual operating costs (including charging costs) and annualized capital costs associated with the TES. This objective function is maximized while adhering to constraints governing plant operation including: a) ramp rates and minimum stable power output of steam turbines, b) capacity constraints that limit operating variables to not exceed design values (e.g. maximum salt flow rate, storage capacity), c) constraints defining energy balance around heat exchanger and heat to power electrical conversion efficiency, d) TES energy storage inventory over time. The key investment decision variables in the model include the sizing of the heat exchangers (economizer, evaporator, superheater), pipes and valves associated with the TES system, molten salt pumps, storage medium and tanks, electric resistive heater (for charging), and, if applicable, the industrial process heater. 

The key techno-economic parameter inputs to the IDS model are summarized in the file "TES_assumptions.xlsx", with remaining parameter inputs summarized in Table S5 and S6 in S.I. In addition to these parameters, for each coal plant archetype, we use the following outputs of the process simulation as parameter inputs for the IDS model: a) log-mean temperature difference (LMTD) for hot and cold composite streams for each heat exchanger, b) steam-cycle heat to power efficiency, and c) maximum salt flow rate. 
The heat exchanger network that are optimized following the design assumptions for a molten salt shell-and-tube heat exchanger network for steam-turbines that consists of a superheater, evaporator, and economizer. 

### Data and Equations
Detailed description of data and equations describing the optimization setup is in: 
[Equations.pdf](https://github.com/serenapatel315/Thermal-Energy-Storage-Retrofit-Model/files/14212777/Equations.pdf)

The Indian Coal Plant Dataset and Simulation Results are found in the folder "aspen_data". The archetypical coal plants are summarized in:
* "archetypes_input.csv": the inputs to the process simulation in aspen
* "arechetypes_output.csv": the outputs to the aspen simulation that were used as a point of comparison between the TES simulation and coal plant simulation 
* "simulation_input_dataset.csv" is the simulation inputs for 85 units as well as the archetype plants
* "simulation_output_dataset.csv" is the simulation results from 85 units as well as the archetype plants
* "output_df_maxtemps.csv" stores the simulation results from the base case (500 MW unit) TES under higher peak salt temperatures

These process simulations and models are stored in the "simulation/Trombay_analysis/Unit 5" directory, which also contains a python notebook with more details on generating sensitivities. 


## How to Use 

### Sample 
The "sample" folder contains the structure for running the optimization models. 
* "clustering parameters" contains .csv files that relate to the clustered electricity price series 
* "parameters" contains the .csv files of parameters for each run 
* "prices" contains the price series in clustered format 
* "results" contains result files, which are written after running the model 

TES_model.jl stores the price-taker model and CEM_model.jl stores a stylized capacity expansion model. Both are written in julia. 
run_model.ipynb provides an example of running the model locally in a julia-based jupyter notebook. 


### Set Parameters
The notebook labeled "set_parameters.ipynb" reads in the parameters of the base case from the TES_Assumptions_2023.xlsx file, as well as data from the fleet. It also sets up the files for each sensitivity. These are stored in the folder labeled "model". 

### Running Cases
After setting up the sensitivity cases of interest, we ran the model on the MIT Supercloud (file paths in TES_repweeks.sh and TES_script_repweeks.jl will need to be changed for personal use), but these can also be run locally as shown in the "sample" files. 

### Processing Results
Results are processed in the file "processing_results.ipynb", which reads in the results from the folder titled "RESULTS". These generate figures stored in the folder labeled "figures". 



