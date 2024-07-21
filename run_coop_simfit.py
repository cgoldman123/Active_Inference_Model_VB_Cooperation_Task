import sys, os, re, subprocess, csv

# this script takes previously fit parameters in fit_results (e.g. fits using MCMC) then runs a matlab script to simulate
# behavior with those parameters, then fit the simulated behavior

fit_results = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/data_analysis/compiled_model_fits_MCMC/coop_MCMC_prolific_7_19_24.csv'
results = sys.argv[1]
experiment_mode = sys.argv[2] # indicate inperson, mturk, or prolific



if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")


ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_CMG/run_coop_simfit.ssub'

with open(fit_results, newline='') as csvfile:
    file = csv.DictReader(csvfile)

    for subject in file:
        stdout_name = f'{results}/logs/{subject["id"]}-%J.stdout'
        stderr_name = f'{results}/logs/{subject["id"]}-%J.stderr'

        jobname = f'coop-fit-{subject["id"]}'
        os.system(f'sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {results} {experiment_mode} {subject["id"]} {subject["mean_alpha"]} {subject["mean_eta"]} {subject["mean_omega"]} {subject["mean_pa"]} {subject["mean_cr"]} {subject["mean_cl"]}') 

        print(f"SUBMITTED JOB [{jobname}]")
        
    


    ###python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_CMG/run_coop_simfit.py  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_VB_model_output/coop_VB_simfit_using_MCMC_params_7-20 "prolific"


    ## joblist | grep coop | grep -Po 98.... | xargs scancel