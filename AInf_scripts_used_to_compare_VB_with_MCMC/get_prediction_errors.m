function pred_errors = get_prediction_errors(subject, fit_results)
num_blocks = fit_results.num_blocks;
num_trials_per_block = fit_results.DCM.MDP.T;
total_trials = num_blocks*num_trials_per_block;

subject_col = repmat(subject, total_trials, 1);
trial_col = (1:total_trials)';

chosen_action_prob_col = nan(total_trials, 1);
G_error_col = nan(total_trials, 1);
outcome_col = nan(total_trials,1);
choice_col = nan(total_trials,1);


for i = 1:num_blocks
    chosen_action_prob = fit_results.MDP_block{i}.chosen_action_probabilities;
    G_error = fit_results.MDP_block{i}.G_error;
    outcome = fit_results.MDP_block{i}.outcomes;
    choice = fit_results.MDP_block{i}.choices;

    % Determine the indices for this block
    idx_start = (i - 1) * 16 + 1;
    idx_end = i * 16;
    
    % Populate columns
    chosen_action_prob_col(idx_start:idx_end) = chosen_action_prob;
    G_error_col(idx_start:idx_end) = G_error;
    outcome_col(idx_start:idx_end) = outcome;
    choice_col(idx_start:idx_end) = choice;

end


% Create the table
pred_errors = table(subject_col, trial_col, chosen_action_prob_col, G_error_col, outcome_col, choice_col,...
    'VariableNames', {'subject', 'trial','chose_action_probability', 'G_error','outcomes','choices'});




end

