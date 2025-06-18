import sys, os, re, subprocess

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/cooperation_prolific_IDs.csv'
results = sys.argv[1]
experiment_mode = sys.argv[2] # indicate inperson, mturk, or prolific


models = [
    {'field': 'alpha,cr,cl,opt,eta,omega','forgetting_split_matrix':0,'learning_split':0},
    {'field': 'alpha,cr,cl,opt,eta_win,eta_loss,eta_neutral,omega','forgetting_split_matrix':0,'learning_split':1},
    {'field': 'alpha,cr,cl,opt,eta,omega_win,omega_loss,omega_neutral','forgetting_split_matrix':1,'learning_split':0},
    {'field': 'alpha,cr,cl,opt,eta,psi_0,psi_r','forgetting_split_matrix':0,'learning_split':0},
    {'field': 'alpha,cr,cl,opt,eta_win,eta_loss,eta_neutral,psi_0,psi_r','forgetting_split_matrix':0,'learning_split':1},
    {'field': 'cr,cl,opt,eta,omega,beta_0,alpha_d','forgetting_split_matrix':0,'learning_split':0},
    {'field': 'cr,cl,opt,eta_win,eta_loss,eta_neutral,omega,beta_0,alpha_d','forgetting_split_matrix':0, 'learning_split':1},
    {'field': 'cr,cl,opt,eta,omega_win,omega_loss,omega_neutral,beta_0,alpha_d','forgetting_split_matrix':1,'learning_split':0}
    ]


if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")

subjects = []
with open(subject_list_path) as infile:
    next(infile)    
    for line in infile:
        subjects.append(line.strip())

ssub_path = '/media/labs/rsmith/lab-members/osanchez/wellbeing/cooperation/scripts/other_models/AI/Final_AI_KPC/run_coop_all_models.ssub'

    
for index, model in enumerate(models, start=1):
    combined_results_dir = os.path.join(results, f"model{index}")
    field = model['field']
    forgetting_split_matrix = model['forgetting_split_matrix']
    learning_split = model['learning_split']

    if not os.path.exists(f"{combined_results_dir}/logs"):
        os.makedirs(f"{combined_results_dir}/logs")
        print(f"Created results-logs directory {combined_results_dir}/logs")
    
    for subject in subjects:
        ssub_path = '/media/labs/rsmith/lab-members/osanchez/wellbeing/cooperation/scripts/other_models/AI/Final_AI_KPC/run_coop_all_models.ssub'
        stdout_name = f"{combined_results_dir}/logs/{subject}-%J.stdout"
        stderr_name = f"{combined_results_dir}/logs/{subject}-%J.stderr"
    
        jobname = f'Coop_model{index}_AI-{subject}'
        os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} \"{subject}\" \"{combined_results_dir}\" \"{field}\" \"{forgetting_split_matrix}\" \"{learning_split}\" \"{experiment_mode}\"")
        #os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject_list} {combined_results_dir} {fit_hierarchical} {field} {drift_mapping} {bias_mapping} {thresh_mapping} {use_parfor} {use_ddm}")
    
        print(f"SUBMITTED JOB [{jobname}]")
    
    ###python3 /media/labs/rsmith/lab-members/osanchez/wellbeing/cooperation/scripts/other_models/AI/Final_AI_KPC/run_coop_all_models.py  /media/labs/rsmith/lab-members/osanchez/wellbeing/cooperation/scripts/other_models/AI/Final_AI_KPC/model_output/ "prolific"


    ## joblist | grep coop | grep -Po 98.... | xargs scancel
    #OR
    ## joblist | grep coop | grep -Po 43... |xargs -n1 scancel