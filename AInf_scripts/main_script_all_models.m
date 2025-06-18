% Main script for model fitting the cooperation task data 
dbstop if error
clear all;

SIM_PASSED_PARAMETERS = false; % this simfits the parameters passed in, instead of simfitting params fit to data
SIMFIT = true;
DO_MODEL_FREE = true;

%%%% WILL HAVE TO CHANGE LOCAL SO THAT WE PROPERLY READ IN LEFT BEHAVIORAL
%%%% FILE

rng(23);
if ispc
    root = 'L:';
    result_dir = [root '/rsmith/lab-members/osanchez/wellbeing/cooperation/model_output/testing/'];
   
    %Modify this w/ the parameters you want to fit
    field = {'cl', 'cr','opt','alpha_d','beta_0','eta_win','eta_loss','eta_neutral','omega'};
    DCM.MDP.forgetting_split_matrix = 0; % separate forgetting rate, matrix-wise
    DCM.MDP.forgetting_split_row = 0; % Separate forgetting rate, rows
    DCM.MDP.learning_split = 1; % 1 = separate wins/losses/neutral, 0 = not

    experiment_mode = "prolific";
    if experiment_mode == "local"
        fit_list = ["BW521"];
    elseif experiment_mode == "prolific"
        fit_list = ["55eb04337480920010aa9e0d"];
    end
    
    simmed_alpha = 3.1454473;
    simmed_cr = 6.9241437;
    simmed_cl = 4.4656905;
    simmed_eta = 0.4853053;
    simmed_omega = 0.46222943;
    simmed_p_a = 0.68625881;
   
elseif isunix
    root = '/media/labs';
    fit_list = string(getenv('SUBJECT'))
    result_dir = getenv('RESULTS')
    field = cellstr(strsplit(getenv('FIELD'),','))
    DCM.MDP.forgetting_split_matrix = str2num(getenv('FORGETTING_SPLIT_MATRIX'))
    DCM.MDP.forgetting_split_row = str2num(getenv('FORGETTING_SPLIT_ROW'))
    DCM.MDP.learning_split = str2num(getenv('LEARNING_SPLIT'))
    experiment_mode = getenv('EXPERIMENT')
end
addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);

% list: BV222, BV696, BW370, BW641

%Fit_file = "./KP123-T1-_COP_R1-_BEH.csv";



for subject = fit_list

    % note that we always fit (even when simulating from parameters passed
    % in, because it sets up the mdp how we need it to run the simulation
    
    DCM.MDP.T = 16;%4; % trials per block

    if DCM.MDP.learning_split == 1
        DCM.MDP.eta_win = .5; %Learning rate
        DCM.MDP.eta_neutral = .5; %Learning rate
        DCM.MDP.eta_loss = .5; %Learning rate
    end 

    if any(contains(field, 'eta'))
        DCM.MDP.eta = .5; %Learning rate 
    else
        DCM.MDP.eta = 1; 
    end

    if DCM.MDP.forgetting_split_matrix == 1 % 1 = separate wins/losses/neutral, 0 = not
        DCM.MDP.omega_win = .25;
        DCM.MDP.omega_loss = .25;
        DCM.MDP.omega_neutral = .25;
    end

    if any(contains(field, 'omega'))
        DCM.MDP.omega = .25; %Forgetting rate 
    else
        DCM.MDP.omega = 0; 
    end

    if any(contains(field, 'psi_r'))
        DCM.MDP.psi_0 = 0.1;
        DCM.MDP.psi_r = 0.25;
        DCM.MDP.dynamic_forgetting = 1;
    else
        DCM.MDP.dynamic_forgetting = 0;
    end
    
    if any(contains(field, 'beta_0'))
        DCM.MDP.beta_0 = 1;
        DCM.MDP.alpha_d = 1;
        DCM.MDP.dynamic_decision_noise = 1;
    else
        DCM.MDP.dynamic_decision_noise = 0;
    end

    
    DCM.MDP.opt = 1/3; %Optimism parameter
    DCM.MDP.cr = 1; %Reward Seeking preference
    DCM.MDP.cl = 1; %Loss aversion
    if ~DCM.MDP.dynamic_decision_noise == 1
        DCM.MDP.alpha = 1; %Action PrQecision/Inverse Temperature
    end
    %Remove fixed variables from DCM.field, leave the ones you want to fit
    DCM.field = field;

    if experiment_mode == "local"
        DCM.MDP.NB = 22;
        [fit_results,file] = TAB_fit_simple_local(subject,DCM);
    elseif experiment_mode == "prolific"
        DCM.MDP.NB = 30;
        %DCM.config.NB = 15; fit only first half of blocks
        [fit_results,file] = TAB_fit_simple_prolific(subject,DCM);
    end
    
    
    if SIM_PASSED_PARAMETERS
        fprintf("FINALLY SIMULATING THE PARAMS. HAD TO FIT FIRST TO GET MDP SET UP\n");
        fit_results.parameters.alpha = simmed_alpha;
        fit_results.parameters.cr = simmed_cr;
        fit_results.parameters.cl = simmed_cl;
        fit_results.parameters.eta = simmed_eta;
        fit_results.parameters.omega = simmed_omega;
        fit_results.parameters.p_a = simmed_p_a;
        
        post_fields = fieldnames(fit_results.prior);
        post_values = struct2cell(fit_results.parameters);
        result_table = cell2table(post_values', 'VariableNames', strcat('simmed_', post_fields));
        result_table.id = subject;
    
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
        post_table.F = fit_results.DCM.F;
        if isfield(fit_results, 'fixed') && ~isempty(fit_results.fixed)
            fixed_fields = fieldnames(fit_results.fixed);
            fixed_values = struct2cell(fit_results.fixed);
            fixed_table = cell2table(fixed_values', 'VariableNames', strcat('fixed_', fixed_fields));
        else
        end
        save([result_dir '/' char(subject) '_fit_results.mat'], "fit_results");
        % Concatenate all tables horizontally
        if isfield(fit_results, 'fixed') && ~isempty(fit_results.fixed)   
            result_table = [addl_vals_table, prior_table, post_table, fixed_table];
        else
            result_table = [addl_vals_table, prior_table, post_table];
            result_table.F = fit_results.DCM.F;
        end
    end

 
    if SIMFIT
        simmed_results = TAB_simfit_simple_prolific(fit_results);
        % assemble output table
        sim_post_fields = fieldnames(simmed_results.prior);
        sim_post_values = struct2cell(simmed_results.parameters);
        sim_post_table = cell2table(sim_post_values', 'VariableNames', strcat('simfit_posterior_', sim_post_fields));
%         sim_post_table.simfit_F = simmed_results.DCM.F;
        sim_addl_vals_table = table(simmed_results.avg_action_prob, simmed_results.model_acc,...
            'VariableNames', {'simfit_avg_action_prob', 'simfit_model_acc'});
        result_table = [result_table, sim_addl_vals_table,  sim_post_table];
        save([result_dir '/' char(subject) '_simfit_results.mat'], "simmed_results");
    end
    
   
    if DO_MODEL_FREE
        mf_results = coop_model_free_20250521(file);
        result_table = [result_table, struct2table(mf_results)];
    end
    
        writetable(result_table, [result_dir '/coop_fit_' char(subject) '.csv']);
    
end


