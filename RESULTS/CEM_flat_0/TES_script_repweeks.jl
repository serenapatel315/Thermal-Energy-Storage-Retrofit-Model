### Execute Functions ### 

println("Loading Inputs")

#include the function
include("TES_model_repweek_CEMFINAL.jl")

#set the path
path = "/home/gridsan/spatel/MITEI_TES/CEM_flat/"

### Read in parameters
dataLoc = "/home/gridsan/spatel/MITEI_TES/CEM_flat/parameters/"
fnames = dataLoc.*readdir(dataLoc)

# Grab the argument that is passed in 
# This is the index to fnames for this process 

task_id = parse(Int,ARGS[1])
num_tasks = parse(Int,ARGS[2])

### Execute Functions ### 

# Check to see if the index is valid (so the program exits cleanly if the wrong indices are passed)
for i in task_id+1:num_tasks:length(fnames)
    println(i)

    println("Solving Model")
    #dictionary of results
    results = TES_dispatch_rep(fnames[i],8,path)
    
    #add in the parameters filename to the dictionary
    #merge!(results, Dict("parameters_filename"=>fnames[i]))
   
    println("Writing Results")
    filename_results = string("results/results_"*string(i)*".csv")
    
    CSV.write(filename_results,results) 
    
#     println("Timing"); 
#     println(@time TES_dispatch_rep!(parameters,8)); 
#     println(@time TES_dispatch_rep!(parameters,8)); 


        
end