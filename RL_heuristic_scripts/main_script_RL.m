% Main script for model fitting the cooperation task data using a Reinforcement learning model
dbstop if error
clear all;

SIM_PASSED_PARAMETERS = false;
SIMFIT = true;
DO_MODEL_FREE = true;

rng(23);
if ispc
    root = 'L:';
    result_dir = [root '/rsmith/lab-members/osanchez/wellbeing/cooperation/model_output/test/'];

    % Modify the names within DCM.field to chose what parameters are going
    % to be fit
    DCM.field = {'beta_0','c','cl','cr','V0','alpha_win','alpha_neutral','alpha_loss','psi','gamma'};
    % Parameter descriptions:
    % beta = inverse temperature; it may change over time, in which case beta_0 is
    % final value of inverse temperature and c determines the rate of
    % change.
    % c = time power constant for dynamic softmax. Positive values of c
    % lead to inverse temperature values that increase over time, while negative values
    % entail inverse temperature values that decrease.
    % cl = loss aversion
    % cr = reward sensitivity
    % V0 = initial value of options
    % alpha = learning rate (may be different for each outcome)
    % psi = forgetting rate (may be different for each outcome)
    % gamma = information bonus
    % eta = determines how quickly the associability weight changes

    experiment_mode = "prolific";
    if experiment_mode == "local"
        fit_list = ["BW521"];
    elseif experiment_mode == "prolific"
        fit_list = ["5afa19a4f856320001cf920f"]; 
    end

    % these are all different from the previous models in coop, since the
    % RL model works differently; Check with Ryan or Carter on what would
    % be the best values to include for these.
    simmed_V0 = 0;
    simmed_alpha = 0;
    simmed_beta = 0;
    simmed_psi = 0;
    simmed_eta = 0;
    simmed_beta_0 = 0;
    simmed_c = 0;
    simmed_gamma = 0;

elseif isunix
    root = '/media/labs';
    fit_list = string(getenv('SUBJECT'))
    result_dir = getenv('RESULTS')
    DCM.field = cellstr(strsplit(getenv('FIELD'),','))
    experiment_mode = string(getenv('EXPERIMENT'))

end

addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);

for subject = fit_list

    DCM.MDP.T = 16; % trials per block

    if any(strcmp(DCM.field,'c'))
        DCM.MDP.softmax = 1; % choose whether or not there will be a dynamic softmax, if so, add beta_0 & c to field DO NOT FIT BETA IF SOFTMAX IS ACTIVE
    else
        DCM.MDP.softmax = 0;
    end
    
    if any(contains(DCM.field, 'psi_win'))
        DCM.MDP.forgetting_split_matrix = 1; % only one forgetting type may be active at once
    else 
        DCM.MDP.forgetting_split_matrix = 0;
    end

    if any(contains(DCM.field, 'alpha_win'))
        DCM.MDP.learning_split = 1;
    else 
        DCM.MDP.learning_split = 0;
    end
    
    if any(strcmp(DCM.field, 'eta'))
        DCM.MDP.assoc = 1; % choose whether or not you want to model associability
    else
        DCM.MDP.assoc = 0;
    end

    if DCM.MDP.learning_split == 1 
        DCM.MDP.alpha_win = 0.5;
        DCM.MDP.alpha_neutral = 0.5;
        DCM.MDP.alpha_loss = 0.5;
        DCM.MDP.alpha = 0.5; %learning rate needs to be passed in for the code to work;
    elseif DCM.MDP.learning_split == 0
        DCM.MDP.alpha = 0.5; % learning rate if not splitting learning 
    end

    if DCM.MDP.forgetting_split_matrix == 1
        DCM.MDP.psi_win = 0.25;
        DCM.MDP.psi_neutral = 0.25;
        DCM.MDP.psi_loss = 0.25;
        DCM.MDP.psi = 0.25;
    elseif DCM.MDP.forgetting_split_matrix == 0
        DCM.MDP.psi = 0.25;
    end

    if DCM.MDP.softmax == 0
        DCM.MDP.beta = 1; % inverse temperature
    elseif DCM.MDP.softmax == 1
        DCM.MDP.beta_0 = 1; % inv. temp. for dynamic softmax
        DCM.MDP.c = 0; % time power constant for dynamic softmax
    end
    if DCM.MDP.assoc == 1 
        DCM.MDP.eta = 0.5; % associability weight
    end
    DCM.MDP.V0 = 0; % initial value
    DCM.MDP.cr = 1; % reward sensitivity
    DCM.MDP.cl = 1; % loss aversion
    DCM.MDP.gamma = 1; % t-t



    if experiment_mode == "local"
        DCM.MDP.NB = 22;
        [fit_results,file] = TAB_fit_simple_local(subject,DCM);
    elseif experiment_mode == "prolific"
        DCM.MDP.NB = 30;
        [fit_results,file] = TAB_fit_simple_prolific(subject,DCM);
    end

if SIM_PASSED_PARAMETERS
        fprintf("FINALLY SIMULATING THE PARAMS. HAD TO FIT FIRST TO GET MDP SET UP\n");
        fit_results.parameters.alpha = simmed_alpha;
        fit_results.parameters.beta = simmed_beta;
        fit_results.parameters.psi = simmed_psi;
        fit_results.parameters.eta = simmed_eta;
        fit_results.parameters.beta_0 = simmed_beta_0;
        fit_results.parameters.c = simmed_c;
        fit_results.parameters.gamma = simmed_gamma;
        fit_results.parameters.cr = simmed_cr;
        fit_results.parameters.cl = simmed_cl;
        fit_results.parameters.V0 = simmed_V0;
        
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