import sys, os, re, subprocess

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/cooperation_prolific_IDs.csv'
#subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/local/local_wellbeing_IDs.csv'

results = sys.argv[1]
experiment_mode = sys.argv[2] # indicate inperson, mturk, or prolific



if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'id' not in line:
            subjects.append(line.strip())

ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_CMG/run_coop.ssub'

for subject in subjects:
    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"

    jobname = f'coop-fit-{subject}'
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {results} {experiment_mode}")

    print(f"SUBMITTED JOB [{jobname}]")
    


    ###python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_CMG/run_coop.py  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_VB_model_output/coop_fits_prolific_8-19-24_one_eta "prolific"
    ###python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_CMG/run_coop.py  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_VB_model_output/coop_fits_local_corrected_8-14-24_three_etas "local"


    ## joblist | grep coop | grep -Po 98.... | xargs scancel