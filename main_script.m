% Main script for model fitting the cooperation task data 
dbstop if error
SIM_PASSED_PARAMETERS = true; % this simfits the parameters passed in, instead of simfitting params fit to data
SIMFIT = true;

%%%% WILL HAVE TO CHANGE LOCAL SO THAT WE PROPERLY READ IN LEFTY BEHAVIORAL
%%%% FILE

rng(23);
if ispc
    root = 'L:';
    result_dir = [root '/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_VB_model_output/'];
    
    experiment_mode = "prolific";
    if experiment_mode == "local"
        fit_list = ["BW521","BV696","BV360"];
    elseif experiment_mode == "prolific"
        fit_list = ["5590a34cfdf99b729d4f69dc"];
    end
    
    simfit_alpha = 3.1454473;
    simfit_cr = 6.9241437;
    simfit_cl = 4.4656905;
    simfit_eta = 0.4853053;
    simfit_omega = 0.46222943;
    simfit_p_a = 0.68625881;
   
elseif isunix
    root = '/media/labs';
    fit_list = string(getenv('SUBJECT'))
    result_dir = getenv('RESULTS')
    experiment_mode = string(getenv('EXPERIMENT'))
    simfit_alpha = str2double(getenv('ALPHA'))
    simfit_cr = str2double(getenv('CR'))
    simfit_cl = str2double(getenv('CL'))
    simfit_eta = str2double(getenv('ETA'))
    simfit_omega = str2double(getenv('OMEGA'))
    simfit_p_a = str2double(getenv('P_A'))

    
    (fit_list)
    (result_dir)
    (experiment_mode)
end
addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);

% list: BV222, BV696, BW370, BW641

%Fit_file = "./KP123-T1-_COP_R1-_BEH.csv";



for subject = fit_list

    % note that we always fit (even when simulating from parameters passed
    % in, because it sets up the mdp how we need it to run the simulation
    
    
    config.forgetting_split = 0; % 1 = separate wins/losses, 0 = not
    config.learning_split = 0; % 1 = separate wins/losses, 0 = not
    config.T = 16; % trials per block

    if experiment_mode == "local"
        config.NB = 22;
        fit_results = TAB_fit_simple_local(subject,config);
    elseif experiment_mode == "prolific"
        config.NB = 30;
        fit_results = TAB_fit_simple_prolific(subject,config);
    end

    
    if SIM_PASSED_PARAMETERS
        fprintf("FINALLY SIMULATING THE PARAMS. HAD TO FIT FIRST TO GET MDP SET UP\n");
        fit_results.parameters.alpha = simfit_alpha;
        fit_results.parameters.cr = simfit_cr;
        fit_results.parameters.cl = simfit_cl;
        fit_results.parameters.eta = simfit_eta;
        fit_results.parameters.omega = simfit_omega;
        fit_results.parameters.p_a = simfit_p_a;
        
        post_fields = fieldnames(fit_results.prior);
        post_values = struct2cell(fit_results.parameters);
        result_table = cell2table(post_values', 'VariableNames', strcat('simmed_', post_fields));
    
    else % fitting to data
        subject
        % assemble output table
        prior_fields = fieldnames(fit_results.prior);
        prior_values = struct2cell(fit_results.prior);
        prior_table = cell2table(prior_values', 'VariableNames', strcat('prior_', prior_fields));
        % Extract additional values
        addl_vals_table = table({char(subject)}, fit_results.file, fit_results.avg_action_prob, fit_results.model_acc, fit_results.has_practice_effects, ...
         'VariableNames', {'id', 'file', 'avg_action_prob', 'model_acc', 'has_practice_effects'});
        post_fields = fieldnames(fit_results.prior);
        post_values = struct2cell(fit_results.parameters);
        post_table = cell2table(post_values', 'VariableNames', strcat('posterior_', post_fields));
        save([result_dir '/' char(subject) '_fit_results.mat'], "fit_results");
        % Concatenate all tables horizontally
        result_table = [addl_vals_table, prior_table, post_table];
    end

 
    if SIMFIT
        simmed_results = TAB_simfit_simple_prolific(fit_results);
        % assemble output table
        sim_post_fields = fieldnames(simmed_results.prior);
        sim_post_values = struct2cell(simmed_results.parameters);
        sim_post_table = cell2table(sim_post_values', 'VariableNames', strcat('simfit_posterior_', sim_post_fields));
        sim_addl_vals_table = table({subject}, simmed_results.avg_action_prob, simmed_results.model_acc,...
            'VariableNames', {'id', 'simfit_avg_action_prob', 'simfit_model_acc'});
        result_table = [sim_addl_vals_table, result_table, sim_post_table];
        save([result_dir '/' char(subject) '_simfit_results.mat'], "simmed_results");
    end
    writetable(result_table, [result_dir '/coop_fit_' char(subject) '.csv']);


   
    
    
    
end


