### Use relevant packages

using JuMP
using Gurobi
using Pandas
using PiecewiseLinearOpt
using CalculusWithJulia
using MathOptInterface
const MOI = MathOptInterface
using CSV
using BenchmarkTools


## MODEL ##
function TES_dispatch_rep(fname,threads,path)
    ## INPUTS ##
        # TES_parameters: DataFrame of the TES parameters 
        # threads: Number of threads that optimization model uses (integer)
        # path: system directory 
    
    ## CREATE PARAMETERS DICTIONARY ##
    TES_parameters = Pandas.read_csv(fname)

    #initialize dictionary
    parameters_dict=Dict()

  
    #set first seven keys to filenames for cluster-related parameters
    parameters_dict["fname_price"]=loc(TES_parameters)[0,"fname_price"]
    parameters_dict["fname_solar"]=loc(TES_parameters)[0,"fname_solar"]
    parameters_dict["fname_wind"]=loc(TES_parameters)[0,"fname_wind"]
    parameters_dict["fname_demand"]=loc(TES_parameters)[0,"fname_demand"]


    parameters_dict["fname_w"]=loc(TES_parameters)[0,"fname_w"]
    parameters_dict["fname_map"]=loc(TES_parameters)[0,"fname_map"]
    parameters_dict["fname_modeled_indices"]=loc(TES_parameters)[0,"fname_modeled_indices"] 

    #for the rest of the columns in TES_parameters dataframe
    for col in columns(TES_parameters)[5+3:end]
        #convert strings to Float64
        if typeof(loc(TES_parameters)[0,col]) == String
            parameters_dict[col]=parse(Float64,loc(TES_parameters)[0,col])
        else
            parameters_dict[col]=loc(TES_parameters)[0,col]
        end
    end

    ### SET PARAMETERS ###
    #interconnection capacity is the rated power
    W_cap = parameters_dict["powerPeak"] #MW

    #system heat loss
    Q_loss = parameters_dict["Q_loss"] #percentage of power input lost in heat transfer

    #aux consumption of TES
    aux_TES = parameters_dict["aux_TES"] #fraction 0 to 1
    
    #hot tank temperature 
    T_hot = parameters_dict["T_hot"] #Kelvin

    #cold tank temperature 
    T_cold = parameters_dict["T_cold"] #Kelvin

    #temperature difference for salt 
    deltaT_salt = T_hot-T_cold

    #heat capacity of salt 
    cp = parameters_dict["cp"] #kJ/kgK = MJ/tonne-K

    #minimum power fraction 
    minpower_fraction = parameters_dict["minpower_fraction"]



    ## ASPEN FEATURES ## 
    #temperature difference at end A of heat exchanger
    deltaT_A = parameters_dict["deltaT_A"]

    #temperature difference at end B of heat exchanger
    deltaT_B = parameters_dict["deltaT_B"]

    #temperature difference at end C of heat exchanger
    deltaT_C = parameters_dict["deltaT_C"]

    #temperature difference at end C of heat exchanger
    deltaT_D = parameters_dict["deltaT_D"]

    #overall log mean temperature difference 
    deltaT_lm = (deltaT_A-deltaT_D) /  log(deltaT_A/deltaT_D)

    #log mean temperature difference for HX A (superheater)
    deltaT_lm_1 = (deltaT_A-deltaT_B) /  log(deltaT_A/deltaT_B) #log is natural log in julia

    # HX 2
    deltaT_lm_2 = (deltaT_B-deltaT_C) /  log(deltaT_B/deltaT_C) 

    # HX 3
    deltaT_lm_3 = (deltaT_C-deltaT_D) /  log(deltaT_C/deltaT_D) 

    #steam cycle efficiency 
    steam_eff=parameters_dict["steam_eff"]; # ASPEN result

    # peak steam flow 
    flowSteam_peak = parameters_dict["flowSteam_peak"]

    # max salt flow 
    flowSalt_max = parameters_dict["flowSalt_max"]

    #heat capacity of steam - from Aspen
    steam_enthalpy = parameters_dict["Q_s"]

    #ratio of heat exchanged in each HX (Q1,Q2,Q3) to total heat exchanged
    ratio_Q1 = parameters_dict["ratio_Q1"]
    ratio_Q2 = parameters_dict["ratio_Q2"]
    ratio_Q3 = parameters_dict["ratio_Q3"]

    ## REP WEEKS PARAMETERS ## 

    # weeks
    W = trunc(Int, parameters_dict["W"]);
     
    # hours per week
    T = trunc(Int, parameters_dict["T"]);

    # number of rep weeks / clusters
    N = trunc(Int, parameters_dict["N"]);

    # read in other cluster parameters
    fname_w = parameters_dict["fname_w"];

    # set cluster parameters to arrays
    w = Array(Pandas.read_csv(path*fname_w))
    fname_map = parameters_dict["fname_map"]

    fn_array = Array(Pandas.read_csv(path*fname_map))
    fname_modeled_indices = parameters_dict["fname_modeled_indices"]

    modeled_indices = Array(Pandas.read_csv(path*fname_modeled_indices))

    
    # COSTS # 
    # tank costs
    cost_tank = parameters_dict["cost_tank"]

    #Fixed Operating and Maintenance cost of TES
    TES_FOM = parameters_dict["TES_FOM"]

    #Variable O&M of TES 
    TES_VOM = parameters_dict["TES_VOM"] #unused

    # cost of charger
    cost_charger = parameters_dict["cost_charger"] #$/MW-t

    # cost of heater for industrial process
    cost_heater = parameters_dict["cost_heater"]

    # heat exchanger A
    c_base_1 = parameters_dict["c_base_1"] #$
    UA_base_1 = parameters_dict["UA_base_1"]; #kJ/s-K 
    n_HX_1 = parameters_dict["n_HX_1"]

    # heat exchanger B
    c_base_2 = parameters_dict["c_base_2"] #$
    UA_base_2 = parameters_dict["UA_base_2"]; #kJ/s-K 
    n_HX_2 = parameters_dict["n_HX_2"]

    # heat exchanger C
    c_base_3 = parameters_dict["c_base_3"] #$
    UA_base_3 = parameters_dict["UA_base_3"]; #kJ/s-K 
    n_HX_3 = parameters_dict["n_HX_3"]

    # UA_max = parameters_dict["UA_max"];
    UA_step = parameters_dict["UA_step"];

    #start up cost incurred per cycle start
    start_cost = parameters_dict["start_cost"]*W_cap; #1k/MW capacity 

    # per unit cost of pipes and valves 
    pipes_valves_cost = parameters_dict["pipes_valves_cost"];

    # parameter to represent any one-time decomissioning cost, retirement costs, or upgrade cost
    upgrade_cost = parameters_dict["upgrade_cost"];

    # subsidies and tax incentives
    gbi_subsidy = parameters_dict["gbi_subsidy"]
    taxcredit = parameters_dict["taxcredit"]

    # energy price profile
    # transform hourly energy price profile into $1k/MWh from $/MWh in repweek format
    fname_price = parameters_dict["fname_price"]
    energyprice_df=Pandas.read_csv(path*"prices/"*fname_price)
    
    cost_energy_t =[]
    for n in 1:N
        vec_n = [iloc(energyprice_df)[n,1:T]./1000]
        cost_energy_t = vcat(vec_n,cost_energy_t)
    end

    # TECHNICAL PARAMETERS #

    #pump parameters   
    hotpump_slope = parameters_dict["hotpump_slope"] 
    hotpump_intercept = parameters_dict["hotpump_intercept"] 

    h = parameters_dict["h"] #15
    g = parameters_dict["g"] #9.81
    pump_eff = parameters_dict["pump_eff"] #0.75

    #cold pump
    coldpump_slope = parameters_dict["coldpump_slope"] #50
    coldpump_intercept = parameters_dict["coldpump_intercept"] 

    #ramp rate 
    ramp_rate_min = parameters_dict["ramp_rate_min"]; #MW/hour

    #charging efficiency for TES 
    beta = parameters_dict["beta"]

    #storage loss factor in tanks (self-discharge)
    alpha_hot = parameters_dict["alpha_hot"]
    
    #big M
    M=parameters_dict["M"]*1000
    
    #Calculate capital recovery factor to annualize capital costs
    life = parameters_dict["life"] #years
    ir = parameters_dict["ir"] #interest rate
    CRF = (ir*(1+ir)^life) / ((1+ir)^life - 1);
    CRF = round(CRF;digits=4)

    #industrial demand percentage
    x =  parameters_dict["x"]; 

    #industrial heat demand as a percentage of peak power
    #Q_ind = x*W_cap/steam_eff #MW_th
    Q_ind = x*W_cap

    #min hours

    min_hours = parameters_dict["min_hours"]

    #UA max set dynamically and round up 
    UA_max = ceil(Int64,W_cap/(steam_eff*deltaT_lm))*2

    ### MODEL SETUP ###
    model = Model(Gurobi.Optimizer);
    set_optimizer_attribute(model, "Threads",threads)


    ### DECISION VARIABLES ###

    # Time-Independent Variables
    @variable(model, 0 <=tank_energy_max <= W_cap*24/steam_eff) ; #MWh 24 hrs. 

    #@variable(model, salt_charge_min <=flowSalt_charge_peak <= flowSalt_max) #tonne/s;

    #@variable(model, 0 <=flowSalt_discharge_peak<=10) #tonne/s;

    @variable(model, 0 <= powerPeak <= W_cap); 

    @variable(model, 0 <= powerTurbine <= W_cap/(1-aux_TES)); 

    @variable(model, 0<=Q_TES_heater_peak<=W_cap/beta) # peak resistive heater for TES  MWh_t

    @variable(model, 0<=Q_IND_heater_peak<=W_cap/beta); #peak industrial heat demand

    @variable(model, 0<=cost_HX) #cost of the HX

    @variable(model, m, Bin)

    # Time-Dependent Variables

    @variable(model,0<=flowSalt_charge_t[1:N,1:T]<=5) #tonne/s - into hot tank

    @variable(model,0<=flowSalt_discharge_t[1:N,1:T]<=5) #tonne/s - out of hot tank

    @variable(model,0<=tank_energy_t[1:N,1:T]<=W_cap*24);# MWh_t

    @variable(model, y[1:N,1:T], Bin, start=0); # 0 (charge) or 1 (discharge)

    @variable(model, 0<=net_charge_t[1:N,1:T] <= W_cap); # net charge power input 

    @variable(model, 0<=net_discharge_t[1:N,1:T] <= W_cap); # net discharge power output 

    @variable(model, 0<=powerTurbine_t[1:N,1:T] <= W_cap/(1-aux_TES)); #power output from turbine

    @variable(model, z_t[1:N,1:T], Bin); #start up decision, binary 0 or 1 (start up decision)

    @variable(model, 0<=q_TES_heater_t[1:N,1:T]) #electric resistive heater for TES 

    @variable(model, 0<=q_heater_t[1:N,1:T]<=W_cap) #electric resistive heater for industry 

    @variable(model, 0<=q_turbind_t[1:N,1:T]) #thermal energy supplied by steam turbine to industry
    
    @variable(model, 0<= q_hx_1[1:N,1:T])
    @variable(model, 0<= q_hx_2[1:N,1:T])
    @variable(model, 0<= q_hx_3[1:N,1:T])

    # Setting piecewise variables with U*area for nonlinear relationship between cost as a function of area
    
    UA_hat_1 = 0:UA_step:UA_max
    UA_hat_2 = 0:UA_step:UA_max
    UA_hat_3 = 0:UA_step:UA_max

    cost_HX_hat_1 = c_base_1*(UA_hat_1/UA_base_1).^n_HX_1
    cost_HX_hat_2 = c_base_2*(UA_hat_2/UA_base_2).^n_HX_2
    cost_HX_hat_3 = c_base_3*(UA_hat_3/UA_base_3).^n_HX_3

    K_1 = length(UA_hat_1)
    K_2 = length(UA_hat_2)
    K_3 = length(UA_hat_3)

    @variable(model, 0 <= UA_1 <= UA_max)
    @variable(model, 0 <= UA_2 <= UA_max)
    @variable(model, 0 <= UA_3 <= UA_max)

    @variable(model, 0<=cost_HX_1)
    @variable(model, 0<=cost_HX_2)
    @variable(model, 0<=cost_HX_3)

    @variable(model, 0 <= λ_1[1:K_1] <= 1);
    @variable(model, 0 <= λ_2[1:K_2] <= 1);
    @variable(model, 0 <= λ_3[1:K_3] <= 1);

    #pump variables
    @variable(model, cost_hotpump>=0)
    @variable(model, cost_coldpump>=0)

    ## representative weeks variables
    @variable(model,deltaQ[1:N])
    @variable(model,Q[1:W]>=0)

    #pump variables
    @variable(model,W_coldpump[1:N,1:T])
    @variable(model,W_hotpump[1:N,1:T])

    @variable(model,0<=P_hotpump)
    @variable(model,0<=P_coldpump)





    ### SET OBJECTIVE FUNCTION ###
    
    @objective(
    model,
    Min,
    (cost_tank*tank_energy_max 
        +pipes_valves_cost*powerTurbine
        +cost_coldpump
        +cost_hotpump
        +cost_HX
        +(cost_charger)*Q_TES_heater_peak/beta
        +cost_heater*Q_IND_heater_peak
        +upgrade_cost*powerTurbine
        )*CRF +TES_FOM*W_cap/(1-aux_TES)
        
        +sum(w[n]* sum(start_cost*z_t[n,t] + cost_energy_t[n][t]*( net_charge_t[n,t]  
        + q_heater_t[n,t]/beta 
        - net_discharge_t[n,t]) - gbi_subsidy*net_discharge_t[n,t] for t in 1:T)
        for n in 1:N )
        );


    ## METRICS ## 
    
    #duration
    #@expression(model, TES_duration, tank_energy_max*steam_eff/powerPeak)

    #dispatch 
    @expression(model, TES_dispatch, sum(w[n]* sum(net_discharge_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(model, TES_charge_dispatch, sum(w[n]* sum(net_charge_t[n,i] for i in 1:T) for n in 1:N ))

    #startup and charge costs 
    @expression(model, TES_charge_costs, sum(w[n]* sum(cost_energy_t[n][i]*net_charge_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(model, TES_startup_costs, sum(w[n]* sum(start_cost*z_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(model, TES_indheater_costs, sum(w[n]* sum(cost_energy_t[n][i]*q_heater_t[n,i]/beta for i in 1:T) for n in 1:N ))
    
    #fixed cost
    @expression(model, TES_FIXED_YR, TES_FOM*W_cap/(1-aux_TES))

    #revenue
    @expression(model, TES_revenue, sum(w[n]* sum(cost_energy_t[n][i]*net_discharge_t[n,i] for i in 1:T) for n in 1:N ))

    #annualized capex 
    @expression(model, TES_annualized_capex,    (cost_tank*tank_energy_max 
    +pipes_valves_cost*powerTurbine
    +cost_coldpump
    +cost_hotpump
    +cost_HX
    +(cost_charger)*Q_TES_heater_peak/beta
    +cost_heater*Q_IND_heater_peak
    +upgrade_cost*powerTurbine)*CRF*(1-taxcredit))

    #capex
    @expression(model, TES_capex, cost_tank*tank_energy_max 
    +pipes_valves_cost*powerTurbine
    +cost_coldpump
    +cost_hotpump
    +cost_HX
    +(cost_charger)*Q_TES_heater_peak
    +cost_heater*Q_IND_heater_peak
    +upgrade_cost*powerTurbine)




    ## CONSTRAINTS ##

    # Net Power Consumption and Production Constraints ----------------------------------
    
    ## net power during charging
    @constraint(model, charge_power[n=1:N,i=1:T], net_charge_t[n,i] == q_TES_heater_t[n,i]/beta + W_coldpump[n,i]/1000);

    ## during discharging
    @constraint(model, discharge_power[n=1:N,i=1:T], net_discharge_t[n,i] == powerTurbine_t[n,i]*(1-aux_TES) - (W_hotpump[n,i]/1000));
    
    #capacity 
    @constraint(model, discharge_powerpeak[n=1:N,i=1:T], net_discharge_t[n,i] <= powerPeak);
            # powerPeak defined <= W_cap (interconnection capacity)

    # Charging mode Constraints --------------------------------------------------------------------
    ## heat supplied during charging process
    @constraint(model, q_heater_TES[n=1:N,i=1:T], q_TES_heater_t[n,i] == flowSalt_charge_t[n,i]*cp*deltaT_salt);

    ## capacity 
    @constraint(model, q_heater_TES_peak[n=1:N,i=1:T], q_TES_heater_t[n,i]<=Q_TES_heater_peak );

    # salt charge flow expression 
    @expression(model, flowSalt_charge_peak, Q_TES_heater_peak/(cp*deltaT_salt))


    #Discharging Mode Constraints --------------------------------------------------------------------
    # Min Power  
    @constraint(model, power_min[n=1:N,i=1:T], powerTurbine_t[n,i] >= y[n,i]*powerTurbine*minpower_fraction);

    # Turbine sizing
    @constraint(model, turbine_size[n=1:N,i=1:T], powerTurbine_t[n,i] <= powerTurbine);
            # powerTurbine defined <= steam turbine capacity (W_cap/(1-aux_TES))

    # Startup constraints for steam cycle                                         # 1 is discharge , 0 is charge
    @constraint(model, starts[n=1:N,i=2:T], z_t[n,i] >= y[n,i]-y[n,i-1])

    # Initialization: period wrapping
    @constraint(model, starts_init[n=1:N,i=1], z_t[n,i] >= y[n,i]-y[n,T]) 
        
    # Ramping Constraints
    # Ramping both directions
    @constraint(model, ramp_rate1[n=1:N,i=2:T], powerTurbine_t[n,i]-powerTurbine_t[n,i-1]<=ramp_rate_min*powerTurbine);
    @constraint(model, ramp_rate2[n=1:N,i=2:T], powerTurbine_t[n,i-1]-powerTurbine_t[n,i]<=ramp_rate_min*powerTurbine);
    # Initial ramp rates wrapping
    @constraint(model, ramp_rate1_init[n=1:N,i=1], powerTurbine_t[n,i]-powerTurbine_t[n,T]<=ramp_rate_min*powerTurbine);
    @constraint(model, ramp_rate2_init[n=1:N,i=1], powerTurbine_t[n,T]-powerTurbine_t[n,i]<=ramp_rate_min*powerTurbine);
    
    #relating turbine power output to salt flow rate
    
    #@constraint(model, steampower_def[n=1:N,i=1:T], powerTurbine_t[n,i]==steam_eff*flowSalt_discharge_t[n,i]*cp*deltaT_salt);
    
    # Salt Discharge flow        
    @expression(model, flowSalt_discharge_peak, powerTurbine/(steam_eff*cp*deltaT_salt))


    ## Mode Switching Constraints --------------------------------------------------------------------
    @constraint(model, flowSalt_charge_constraint[n=1:N,i=1:T], flowSalt_charge_t[n,i] <= M*(1-y[n,i]))
    
    @constraint(model, flowSalt_discharge_constraint[n=1:N,i=1:T], flowSalt_discharge_t[n,i] <= M*y[n,i]);
    

    ## Salt Pump Constraints --------------------------------------------------------------------
    # Consumption from pumps 
    @constraint(model, P_hp[n=1:N,i=1:T], W_hotpump[n,i] == flowSalt_discharge_t[n,i]*h*g/pump_eff ); #kW
    @constraint(model, P_cp[n=1:N,i=1:T], W_coldpump[n,i] == flowSalt_charge_t[n,i]*h*g/pump_eff ); #kW
        
    # Capacity of the pumps
    @constraint(model, P_hpsize[n=1:N,i=1:T], W_hotpump[n,i] <= P_hotpump ); #kW
    @constraint(model, P_cpsize[n=1:N,i=1:T], W_coldpump[n,i] <= P_coldpump); #kW
    
    # Cost of the pumps 
    @constraint(model, cost_coldpump_def, cost_coldpump==(P_coldpump*(coldpump_slope)+coldpump_intercept*m))
    @constraint(model, cost_hotpump_def, cost_hotpump ==(P_hotpump*(hotpump_slope)+hotpump_intercept*m))

    # Binary variable constraint for pumps
    @constraint(model, bin_hotpump, P_hotpump <= m*M)
    @constraint(model, bin_coldpump, P_coldpump <= m*M)


    ## Heat Exchanger Network Design Constraints --------------------------------------------------------------------
    ## total thermal energy exchanged in heat exchanger network 
    @constraint(model, hx_balance_t[n=1:N,i=1:T], q_hx_1[n,i]+q_hx_2[n,i]+q_hx_3[n,i] == flowSalt_discharge_t[n,i]*cp*deltaT_salt);
    
    ## Cannot exceed capacity
    @constraint(model, hx_1_balance[n=1:N,i=1:T], q_hx_1[n,i] <= UA_1 * deltaT_lm_1);
    @constraint(model, hx_2_balance[n=1:N,i=1:T], q_hx_2[n,i] <= UA_2 * deltaT_lm_2);
    @constraint(model, hx_3_balance[n=1:N,i=1:T], q_hx_3[n,i] <= UA_3 * deltaT_lm_3);

    ## Cannot exceed ratio between heat exchangers
    @constraint(model, Q_hx_1_ratio[n=1:N,i=1:T], q_hx_1[n,i] >= ratio_Q1*(q_hx_1[n,i]+q_hx_2[n,i]+q_hx_3[n,i]));
    @constraint(model, Q_hx_2_ratio[n=1:N,i=1:T], q_hx_2[n,i] >= ratio_Q2*(q_hx_1[n,i]+q_hx_2[n,i]+q_hx_3[n,i]));
    @constraint(model, Q_hx_3_ratio[n=1:N,i=1:T], q_hx_3[n,i] >= ratio_Q3*(q_hx_1[n,i]+q_hx_2[n,i]+q_hx_3[n,i]));


    ## Total Cost of heat exchanger network 
    @constraint(model, hx_cost_total, cost_HX >= cost_HX_1+cost_HX_2+cost_HX_3);
    
    ## Piecwise Linearization of costs for each HX
    
    @constraint(model, cost_HX_1 == sum(cost_HX_hat_1[i] * λ_1[i] for i=1:K_1))
    @constraint(model, cost_HX_2 == sum(cost_HX_hat_2[i] * λ_2[i] for i=1:K_2))
    @constraint(model, cost_HX_3 == sum(cost_HX_hat_3[i] * λ_3[i] for i=1:K_3))

    @constraint(model, UA_1 == sum(UA_hat_1[i] * λ_1[i] for i=1:K_1))
    @constraint(model, UA_2 == sum(UA_hat_2[i] * λ_2[i] for i=1:K_2))
    @constraint(model, UA_3 == sum(UA_hat_3[i] * λ_3[i] for i=1:K_3))

    @constraint(model, sum(λ_1) == 1)
    @constraint(model, sum(λ_2) == 1)
    @constraint(model, sum(λ_3) == 1)

    @constraint(model, λ_1 in SOS2())
    @constraint(model, λ_2 in SOS2())
    @constraint(model, λ_3 in SOS2())


    ## Energy Storage Constraints --------------------------------------------------------------------
        ## Energy Storage Balance
    @constraint(model, tank_inventory[n=1:N,i=2:T], tank_energy_t[n,i] == tank_energy_t[n,i-1] + 
        (flowSalt_charge_t[n,i] - flowSalt_discharge_t[n,i])*cp*(T_hot-T_cold) - alpha_hot*tank_energy_t[n,i-1])

        ## Initialize storage size to be equal to storage energy at the end of period 
    @constraint(model, tank_energy_initialize[n=1:N,i=1], tank_energy_t[n,i]==(1-alpha_hot)*(tank_energy_t[n,T]-deltaQ[n])+
        (flowSalt_charge_t[n,i] - flowSalt_discharge_t[n,i])*cp*(T_hot-T_cold));

        ## Energy Storage Capacity 
    @constraint(model, tank_energy_constraint[n=1:N,i=1:T], tank_energy_t[n,i]<=tank_energy_max)
    
        ## Min duration powerPeak_min*min_hours / steam_eff 
    @constraint(model, tank_energy_duration, W_cap*min_hours/steam_eff <= tank_energy_max)
    
    # if min_hours > 0 
    #     @constraint(model, power_min_hours, W_cap/5 <= powerTurbine)
    # end

    # Representative week constraints --------------------------------------------------------------------
        ## Storage inventory for modeled period
    @constraint(model,Q_inventory[w=2:W], Q[w] == (1-(alpha_hot*24*7))*Q[w-1] + deltaQ[fn_array[w]+1])

        ## First Modeled period 
    @constraint(model,Q_inventory_init, Q[1] == (1-(alpha_hot*24*7))*Q[W] + deltaQ[fn_array[W]+1])

        ## Interexchange
    @constraint(model,Q_inter[n=modeled_indices], Q[n+1] == tank_energy_t[fn_array[n+1]+1,T] - deltaQ[fn_array[n+1]+1]) 
    @constraint(model,Q_max[w=1:W], Q[w] <= tank_energy_max)
    
        ## Additional constraint because unusually high intraweek variability (max and min SOC)    
     @constraint(model,SOC_max[w=1:W,i=2:T], Q[w] + tank_energy_t[fn_array[w]+1,i-1] + 
    (flowSalt_charge_t[fn_array[w]+1,i] - flowSalt_discharge_t[fn_array[w]+1,i])*cp*(T_hot-T_cold) - alpha_hot*tank_energy_t[fn_array[w]+1,i-1] - tank_energy_t[fn_array[w]+1,1] <= tank_energy_max)
    
    @constraint(model,SOC_min[w=1:W,i=2:T], Q[w] + tank_energy_t[fn_array[w]+1,i-1] + 
    (flowSalt_charge_t[fn_array[w]+1,i] - flowSalt_discharge_t[fn_array[w]+1,i])*cp*(T_hot-T_cold) - alpha_hot*tank_energy_t[fn_array[w]+1,i-1] - tank_energy_t[fn_array[w]+1,1] >= 0)
  

    ### INDUSTRIAL CO-PROCESS CONSTRAINTS ### --------------------------------------------------------------------
    
        ##Power Definition
    @constraint(model, industrial_power[n=1:N,i=1:T],  powerTurbine_t[n,i] == steam_eff*(flowSalt_discharge_t[n,i]*cp*deltaT_salt - q_turbind_t[n,i]))
    #@constraint(model, industrial_power[n=1:N,i=1:T],  powerTurbine_t[n,i] == steam_eff*(flowSalt_discharge_t[n,i]*cp*deltaT_salt))

        ## Industrial heat is supplied either by the electric resistive heater or the steam when in discharge mode
    @constraint(model, heat_supply[n=1:N,i=1:T], Q_ind == q_heater_t[n,i] + q_turbind_t[n,i])
    
        ## Maximum size of resistive heater for local heat
    @constraint(model, heater_size[n=1:N,i=1:T], q_heater_t[n,i] <= Q_IND_heater_peak)

        ## Q_indsteam can only be active when in discharge mode
    @constraint(model, Q_steam[n=1:N,i=1:T], q_turbind_t[n,i] <= y[n,i]*Q_ind)
    
        ## Metrics for q_turbine_t dispatch and q_heater_t dispatch
    @expression(model, q_heater_dispatch,sum(w[n]* sum(q_heater_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(model, q_turbind_dispatch,sum(w[n]* sum(q_turbind_t[n,i] for i in 1:T) for n in 1:N ))
    

    ### SET OPTIMIZER ATTRIBUTES ###
    set_optimizer_attribute(model, "MIPGap", 4e-2) #default is 1e-4
    set_optimizer_attribute(model, "TimeLimit", 60*30) #default is unlimited seconds
    set_optimizer_attribute(model, "LogFile", "gurobi_log.txt")

    ### OPTIMIZE MODEL ### 
    print(fname)
    optimize!(model)
    #print(model)
    print(solution_summary(model, verbose=true))

    # if termination_status(model) == OPTIMAL
    #     println("Solution is optimal")
    # elseif termination_status(model) == TIME_LIMIT && has_values(model)
    #     println("Solution is suboptimal due to a time limit, but a primal solution is available")
    # else
    #     error("The model was not solved correctly.")
    # end
    # println("  objective value = ", objective_value(model))
    # if primal_status(model) == FEASIBLE_POINT
    #     println("  primal solution: x = ", value(x))
    # end
    # if dual_status(model) == FEASIBLE_POINT
    #     println("  dual solution: c1 = ", dual(c1))
    # end

    #initialize
    TES_LCOE_dict = Dict()
    TES_LCOE_charge_costs = 0
    TES_LCOE_startup_costs = 0
    TES_LCOE_indheater_costs = 0
    TES_LCOE_revenue = 0
    TES_LCOE_capex = 0
    TES_LCOE_profit = 0
    TES_LCOE_FOM_costs = 0

    #LCOE dataframe - TES_dispatch needs to be nonzero for the rest to be calculated... 
    TES_dispatch = value.(model[:TES_dispatch])
    #this is where the error gets through 
    if TES_dispatch > 0
        
        #create the expressions
        TES_LCOE_charge_costs = value.(model[:TES_charge_costs])/TES_dispatch
        TES_LCOE_startup_costs = value.(model[:TES_startup_costs])/TES_dispatch
        TES_LCOE_indheater_costs = value.(model[:TES_indheater_costs])/TES_dispatch
        TES_LCOE_FOM_costs = value.(model[:TES_FIXED_YR])/TES_dispatch
        TES_LCOE_revenue = value.(model[:TES_revenue])/TES_dispatch
        TES_LCOE_capex = value.(model[:TES_annualized_capex])/TES_dispatch
        TES_LCOE_profit = objective_value(model)/TES_dispatch


        #make the dictionary 
        TES_LCOE_dict = Dict("Startup" => TES_LCOE_startup_costs, 
                        "Charge" => TES_LCOE_charge_costs,
                        "IndustrialCharge" => TES_LCOE_indheater_costs, 
                        "Revenue" => TES_LCOE_revenue, 
                        "Profits" => TES_LCOE_profit, 
                        "Capex" => TES_LCOE_capex, 
                        "total" => TES_LCOE_startup_costs+TES_LCOE_charge_costs+TES_LCOE_indheater_costs+TES_LCOE_capex
        )

    end

    
    

    ### SIMPLE LI-ION RUN UNDER SAME PRICE PROFILE ### ------------------------------

    # parameters --------------------------------------------------------------------
    #investment cost of generation per MW ($1k/year)
    a_inv_p = parameters_dict["a_inv_p"] #48.8945 #
    a_inv_e = parameters_dict["a_inv_e"] #17.3089

    #fixed O&M $1k/MW/year 
    a_fom_p = parameters_dict["a_fom_p"]  #6009 / 1000 a_fom_e
    a_fom_e = 0.025*a_inv_e

    #coperating costs ($1k) per MWh
    a_opex_p = parameters_dict["a_opex_p"]  # 0.1/1000 #$1k/MWh 
    a_opex_c = 1/1000 #$1/MWh charge 

    #storage efficiencies
    eta_self = parameters_dict["eta_self"] #0
    eta_charge = parameters_dict["eta_charge"] #0.95
    eta_discharge = parameters_dict["eta_discharge"] #0.95

    ### MODEL SETUP ### --------------------------------------------------------------------
    BESS_model = Model(Gurobi.Optimizer);
    set_optimizer_attribute(BESS_model, "Threads",threads)
    
    # decision variables --------------------------------------------------------------------
    @variable(BESS_model,0<=capP_Li<=W_cap)
    @variable(BESS_model,0<=capE_Li)

    @variable(BESS_model,0<=a_dis_t[1:N,1:T])
    @variable(BESS_model,0<=a_ch_t[1:N,1:T])
    @variable(BESS_model,0<=soc_t[1:N,1:T])

    # objective function --------------------------------------------------------------------
    @objective(
        BESS_model,
        Min,
        a_fom_p*capP_Li + a_fom_e*capE_Li +
        a_inv_p*capP_Li + a_inv_e*capE_Li + 
            +sum(w[n]* sum(a_opex_p*a_dis_t[n,t] 
            + cost_energy_t[n][t]*(a_ch_t[n,t]- a_dis_t[n,t]) for t in 1:T)
            for n in 1:N )
            );

    # expression for duration
    #@expression(BESS_model, Li_duration, capE_Li*eta_discharge/capP_Li)

    # constraints --------------------------------------------------------------------
    @expression(BESS_model, Li_dispatch, sum(w[n]* sum(a_dis_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(BESS_model, Li_revenue, sum(w[n]* sum(cost_energy_t[n][i]*a_dis_t[n,i] for i in 1:T) for n in 1:N ))

    ## Li-ion Energy Balance and Inventory Constraint 
    @constraint(BESS_model, Li_inventory[n=1:N,i=2:T], soc_t[n,i] == soc_t[n,i-1] + 
    a_ch_t[n,i]*eta_charge - a_dis_t[n,i]/eta_discharge - eta_self*soc_t[n,i-1])

    #Initialize 
    @constraint(BESS_model, Li_energy_initialize[n=1:N,i=1], soc_t[n,i]==(1-eta_self)*(soc_t[n,T])+
    a_ch_t[n,i]*eta_charge - a_dis_t[n,i]/eta_discharge);

    #charge 
    @constraint(BESS_model, cap_limit_charge[n=1:N,i=1:T],a_dis_t[n,i] <= capP_Li)
    @constraint(BESS_model, cap_limit_discharge[n=1:N,i=1:T],a_ch_t[n,i] <= capP_Li)
    @constraint(BESS_model, cap_limit_energy[n=1:N,i=1:T],soc_t[n,i] <= capE_Li)

    #min duration
    @constraint(BESS_model, min_duration, min_hours*W_cap <= capE_Li)
    # if min_hours > 0
    #     @constraint(BESS_model, power_min_hours, W_cap/5 <= capP_Li)
    # end
    

    ### SET OPTIMIZER ATTRIBUTES ### --------------------------------------------------------------------
    set_optimizer_attribute(BESS_model, "MIPGap", 4e-2) #default is 1e-4
    set_optimizer_attribute(BESS_model, "TimeLimit", 60*30) #default is unlimited seconds
    set_optimizer_attribute(BESS_model, "LogFile", "gurobi_log.txt")
    
    ### OPTIMIZE MODEL ### 
    optimize!(BESS_model)
    #print(model)
    print(solution_summary(BESS_model, verbose=true))

    # results --------------------------------------------------------------------
    # LCOE calculation

    Li_LCOE_dict = Dict()

    #initialize
    Li_LCOE_charge_costs = 0
    Li_LCOE_opex_costs = 0
    Li_LCOE_fom_costs = 0
    Li_LCOE_revenue = 0
    Li_LCOE_capex = 0
    Li_LCOE_profit = 0

    Li_dispatch = value.(BESS_model[:Li_dispatch])

    if Li_dispatch > 0

        @expression(BESS_model, Li_charge_costs, sum(w[n]* sum(cost_energy_t[n][i]*a_ch_t[n,i] for i in 1:T) for n in 1:N ))
        @expression(BESS_model, Li_opex_costs, sum(w[n]* sum(a_opex_p*a_dis_t[n,i] + a_opex_c*a_ch_t[n,i]  for i in 1:T) for n in 1:N ))
        @expression(BESS_model, Li_FOM, a_fom_p*capP_Li + a_fom_e*capE_Li)
        @expression(BESS_model, Li_INV, a_inv_p*capP_Li + a_inv_e*capE_Li)
      
        #create the expressions
        Li_LCOE_charge_costs = value.(BESS_model[:Li_charge_costs])/Li_dispatch
        Li_LCOE_opex_costs = value.(BESS_model[:Li_opex_costs])/Li_dispatch
        Li_LCOE_fom_costs = value.(BESS_model[:Li_FOM])/Li_dispatch
        Li_LCOE_revenue = value.(BESS_model[:Li_revenue])/Li_dispatch
        Li_LCOE_capex = value.(BESS_model[:Li_INV])/Li_dispatch
        Li_LCOE_profit = objective_value(BESS_model)/Li_dispatch


        #make the dictionary 
        Li_LCOE_dict = Dict(
                        "Charge" => Li_LCOE_charge_costs,
                        "Opex"=>Li_LCOE_opex_costs,
                        "Fom" =>Li_LCOE_fom_costs,
                        "Revenue" => Li_LCOE_revenue, 
                        "Profits" => Li_LCOE_profit, 
                        "Capex" => Li_LCOE_capex, 
                        "total" => Li_LCOE_charge_costs+Li_LCOE_opex_costs+Li_LCOE_fom_costs+Li_LCOE_capex
        )
    end



    



    ### SIMPLE COAL PLANT DISPATCH### ------------------------------

    # parameters --------------------------------------------------------------------
    coal_fuel_cost = parameters_dict["coal_fuel_cost"]
    coal_VOM = parameters_dict["coal_VOM"]
    start_up_fuel = parameters_dict["start_up_fuel"]*W_cap
    coal_start_cost = start_up_fuel*coal_fuel_cost + start_cost #$1k/start

    c_inv_coal = parameters_dict["c_inv_coal"] #0 #111.835 assumes no investment cost
    c_fom_coal = parameters_dict["c_fom_coal"]  #31248 / 1000 
    c_opex_coal = coal_VOM+coal_fuel_cost #marginal cost #30/1000 

    coal_emf = parameters_dict["coal_emf"] #0.5 # tCO2/MWh parameters_dict["emissions_rate"]

    ### MODEL SETUP ### --------------------------------------------------------------------
    coal_model = Model(Gurobi.Optimizer);
    set_optimizer_attribute(coal_model, "Threads",threads)
    
    # decision variables --------------------------------------------------------------------
    @variable(coal_model,0<=capCoal<=W_cap)
    @variable(coal_model,0<=p_coal_t[1:N,1:T])
    @variable(coal_model, y_coal[1:N,1:T], Bin, start=0); # coal tracking power
    @variable(coal_model, z_coal_t[1:N,1:T], Bin)


    # objective function --------------------------------------------------------------------
    @objective(
    coal_model,
    Min,
    c_fom_coal*capCoal +
    c_inv_coal*capCoal
        +sum(w[n]* sum(coal_start_cost*z_coal_t[n,t] + c_opex_coal*p_coal_t[n,t] - cost_energy_t[n][t]*p_coal_t[n,t] for t in 1:T)
        for n in 1:N )
        );


    # constraints --------------------------------------------------------------------
    ## Emissions constraint --------------------------------------------------------------------
    # BaselineEmissions = parameters_dict["BaselineEmissions"] #6 #tCO2 total 
    # emissions_percent = parameters_dict["emissions_percent"] #1
    
    #track emissions
    @expression(coal_model, emissions, sum(w[n]* sum(coal_emf*p_coal_t[n,i] for i in 1:T) for n in 1:N ))
    
    # add emissions constraint if activated
    # if BaselineEmissions > 0
    #     @constraint(coal_model,emissions_cap, emissions <= emissions_percent*BaselineEmissions)
    # end

    #sizing
    @constraint(coal_model, cap_limit_coal[n=1:N,i=1:T],p_coal_t[n,i] <= capCoal*y_coal[n,i])
    
    #min power 
    @constraint(coal_model, power_min_coal[n=1:N,i=1:T], p_coal_t[n,i] >= y_coal[n,i]*capCoal*minpower_fraction);

    # Ramp rates
    @constraint(coal_model, ramp_rate_coal_1[n=1:N,i=2:T], p_coal_t[n,i]-p_coal_t[n,i-1]<=ramp_rate_min*capCoal);
    @constraint(coal_model, ramp_rate_coal_2[n=1:N,i=2:T], p_coal_t[n,i-1]-p_coal_t[n,i]<=ramp_rate_min*capCoal);
    
    # #initial ramp rates wrapping
    @constraint(coal_model, ramp_rate1_coal_init[n=1:N,i=1], p_coal_t[n,i]-p_coal_t[n,T]<=ramp_rate_min*capCoal);
    @constraint(coal_model, ramp_rate2_coal_init[n=1:N,i=1], p_coal_t[n,T]-p_coal_t[n,i]<=ramp_rate_min*capCoal);
    
    # Start up 
    @constraint(coal_model, starts_coal[n=1:N,i=2:T], z_coal_t[n,i] >= y_coal[n,i]-y_coal[n,i-1]) 
    # # first startup constraint in each period wrapping
    @constraint(coal_model, starts_init_coal[n=1:N,i=1], z_coal_t[n,i] >= y_coal[n,i]-y_coal[n,T])
    

    # execute model
    set_optimizer_attribute(coal_model, "MIPGap", 4e-2) #default is 1e-4
    set_optimizer_attribute(coal_model, "TimeLimit", 60*30) #default is unlimited seconds
    set_optimizer_attribute(coal_model, "LogFile", "gurobi_log.txt")

    ### OPTIMIZE MODEL ### 
    optimize!(coal_model)
    #print(model)
    print(solution_summary(coal_model, verbose=true))

# results --------------------------------------------------------------------
    @expression(coal_model, coal_dispatch, sum(w[n]* sum(p_coal_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(coal_model, coal_startup_costs, sum(w[n]* sum(coal_start_cost*z_coal_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(coal_model, coal_opex_costs, sum(w[n]* sum(c_opex_coal*p_coal_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(coal_model, coal_revenue, sum(w[n]* sum(cost_energy_t[n][i]*p_coal_t[n,i] for i in 1:T) for n in 1:N ))
    @expression(coal_model, coal_FOM, c_fom_coal*capCoal)
    @expression(coal_model, coal_INV, c_inv_coal*capCoal)
    # LCOE calculation

    #initialize coal 
    coal_LCOE_startup_costs = 0
    coal_LCOE_opex_costs = 0
    coal_LCOE_fom_costs = 0
    coal_LCOE_revenue = 0
    coal_LCOE_capex = 0
    coal_LCOE_profit = 0

    coal_dispatch = value.(coal_model[:coal_dispatch])
    if coal_dispatch > 0
      
        #create the expressions
        coal_LCOE_startup_costs = value.(coal_model[:coal_startup_costs])/coal_dispatch
        coal_LCOE_opex_costs = value.(coal_model[:coal_opex_costs])/coal_dispatch
        coal_LCOE_fom_costs = value.(coal_model[:coal_FOM])/coal_dispatch
        coal_LCOE_revenue = value.(coal_model[:coal_revenue])/coal_dispatch
        coal_LCOE_capex = value.(coal_model[:coal_INV])/coal_dispatch
        coal_LCOE_profit = objective_value(coal_model)/coal_dispatch

    end


    
    ### WRITE ALL OUTPUTS TO DICTIONARY ### 

    output_dict = Dict("tank_energy_t"=>vec(value.(model[:tank_energy_t])),
    "flowSalt_discharge_t"=>vec(value.(model[:flowSalt_discharge_t])),
    "flowSalt_charge_t"=>vec(value.(model[:flowSalt_charge_t])),

    "y_binary"=>vec(value.(model[:y])),
    "tank_energy_max"=> value.(model[:tank_energy_max]),
    "flowSalt_charge_peak"=> value.(model[:flowSalt_charge_peak]),
    "flowSalt_discharge_peak"=> value.(model[:flowSalt_discharge_peak]),

    "UA_1"=> value.(model[:UA_1]),
    "UA_2"=> value.(model[:UA_2]),
    "UA_3"=> value.(model[:UA_3]),

    "cost_HX_1"=> value.(model[:cost_HX_1]),
    "cost_HX_2"=> value.(model[:cost_HX_2]),
    "cost_HX_3"=> value.(model[:cost_HX_3]),

    "cost_HX"=> value.(model[:cost_HX_1])+value.(model[:cost_HX_2])+value.(model[:cost_HX_3]),
    
    "cost_energy_t"=>vec(cost_energy_t),
    "objective_value"=>objective_value(model),

    "net_charge_t"=> vec(value.(model[:net_charge_t])),
    "net_discharge_t"=> vec(value.(model[:net_discharge_t])),

    "powerPeak"=> value.(model[:powerPeak]),
    "powerTurbine"=> value.(model[:powerTurbine]),
    "powerTurbine_t"=> vec(value.(model[:powerTurbine_t])),


    "start_z_t"=> vec(value.(model[:z_t])),
    "deltaQ"=> vec(value.(model[:deltaQ])),
    "Q"=> vec(value.(model[:Q])),
    "charger_cost"=>cost_charger*(value.(model[:flowSalt_charge_peak])*cp*deltaT_salt),
    "cost_tank"=>cost_tank*value.(model[:tank_energy_max]),
    "pipes_valves_cost"=>pipes_valves_cost*value.(model[:powerTurbine]),
    "cost_coldpump"=>value.(model[:cost_coldpump]),
    "cost_hotpump"=>value.(model[:cost_hotpump]),
    "CRF"=>CRF,
    "subsidy"=>gbi_subsidy,
    "weights"=>w,
    "modeled_indices" => modeled_indices,
    "cp"=> cp, 
    "deltaT_salt"=> deltaT_salt,
    "steam_eff"=> steam_eff,
    "fn_map"=> fn_array,
    "runtime"=> solve_time(model),
    "lifetime" => life, 
    "upgrade_cost" => upgrade_cost,
    "price_fname" => fname_price,
    "q_heater_t" => vec(value.(model[:q_heater_t])),
    "Q_IND_heater_peak" => value.(model[:Q_IND_heater_peak]),
    "Q_TES_heater_peak" => value.(model[:Q_TES_heater_peak]),
    "q_turbind_t" => vec(value.(model[:q_turbind_t])),
    "Q_ind"=>Q_ind,
    "P_hotpump" => value.(model[:P_hotpump]),
    "P_coldpump" => value.(model[:P_coldpump]),
    "relative_gap" => relative_gap(model),
    "fname" => fname,
    "lambda_1"=> vec(value.(model[:λ_1])),
    "lambda_2"=> vec(value.(model[:λ_2])),
    "lambda_3"=> vec(value.(model[:λ_3])),
    "min_hours"=> min_hours,

    "q_heater_dispatch" =>value.(model[:q_heater_dispatch]),
    "q_turbind_dispatch" => value.(model[:q_turbind_dispatch]),


    "m"=> value.(model[:m]),
    "y_binary_coal"=>vec(value.(coal_model[:y_coal])),


    "CAPEX_TES_storage" => cost_tank*value.(model[:tank_energy_max]) , 
    "CAPEX_TES_pipes_valves" => pipes_valves_cost*value.(model[:powerTurbine]), 
    "CAPEX_TES_pumps" => value.(model[:cost_coldpump]) + value.(model[:cost_hotpump]), 
    "CAPEX_TES_HEN" => value.(model[:cost_HX]) ,
    "CAPEX_TES_charger" => (cost_charger)*value.(model[:Q_TES_heater_peak]) ,
    "CAPEX_TES_Ind_charger" => cost_heater*value.(model[:Q_IND_heater_peak]) ,
    "CAPEX_TES_upgrade_costs" => upgrade_cost*value.(model[:powerTurbine]) ,
    "CAPEX_TES_total" => value.(model[:TES_capex]) ,
    
    "TES_LCOE_dict" => TES_LCOE_dict,
    "TES_LCOE_Startup" => TES_LCOE_startup_costs, 
    "TES_LCOE_Charge" => TES_LCOE_charge_costs,
    "TES_LCOE_IndustrialCharge" => TES_LCOE_indheater_costs, 
    "TES_LCOE_Revenue" => TES_LCOE_revenue, 
    "TES_LCOE_Profits" => TES_LCOE_profit, 
    "TES_LCOE_Capex" => TES_LCOE_capex, 
    "TES_LCOE_FOM_costs" => TES_LCOE_FOM_costs,
    "TES_LCOE_total" => TES_LCOE_FOM_costs + 
                        TES_LCOE_startup_costs + 
                        TES_LCOE_charge_costs +
                        TES_LCOE_indheater_costs +
                        TES_LCOE_capex,

    "TES_dispatch" => value.(model[:TES_dispatch]) ,
    "TES_charge_dispatch" => value.(model[:TES_charge_dispatch]) ,
    "TES_charge_costs" => value.(model[:TES_charge_costs]) ,
    "TES_startup_costs" => value.(model[:TES_startup_costs]) ,
    "TES_indheater_costs" => value.(model[:TES_indheater_costs]) ,
    "TES_revenue" => value.(model[:TES_revenue]) ,
    "TES_annualized_capex" => value.(model[:TES_annualized_capex]) ,
    "TES_capex" => value.(model[:TES_capex]) ,
    "TES_fom_yr"=> value.(model[:TES_FIXED_YR]),
    
    "TES_duration" => steam_eff*value.(model[:tank_energy_max])/value.(model[:powerTurbine]) ,
    # ind dispatch 

    "Li_LCOE_dict" => Li_LCOE_dict,
    "Li_objective" => objective_value(BESS_model),
    "Li_LCOE_Charge" => Li_LCOE_charge_costs,
    "Li_LCOE_Opex"=>Li_LCOE_opex_costs,
    "Li_LCOE_Fom" =>Li_LCOE_fom_costs,
    "Li_LCOE_Revenue" => Li_LCOE_revenue, 
    "Li_LCOE_Profits" => Li_LCOE_profit, 
    "Li_LCOE_Capex" => Li_LCOE_capex, 
    "Li_LCOE_total" => Li_LCOE_charge_costs+Li_LCOE_opex_costs+Li_LCOE_fom_costs+Li_LCOE_capex,
    "Li_dispatch" => value.(BESS_model[:Li_dispatch]) ,
    "Li_duration" => value.(BESS_model[:capE_Li])*eta_discharge/value.(BESS_model[:capP_Li]) ,


    "Li_net_charge_t"=> vec(value.(BESS_model[:a_ch_t])),
    "Li_net_discharge_t"=> vec(value.(BESS_model[:a_dis_t])),
    "capE_Li" => value.(BESS_model[:capE_Li]),
    "capP_Li" => value.(BESS_model[:capP_Li]),
    "soc_t" => vec(value.(BESS_model[:soc_t])),
    
    "coal_objective" => objective_value(coal_model),
    "coal_LCOE_startup_costs" => coal_LCOE_startup_costs, 
    "coal_LCOE_opex_costs" => coal_LCOE_opex_costs,
    "coal_LCOE_fom_costs" => coal_LCOE_fom_costs,
    "coal_LCOE_revenue" => coal_LCOE_revenue,
    "coal_LCOE_capex" => coal_LCOE_capex,
    "coal_LCOE_profit" => coal_LCOE_profit,
    "coal_dispatch" => value.(coal_model[:coal_dispatch]) ,



    )
    
     
    return output_dict
end
