% Function for running the basic Rescorla-Wagner reinforcement learning
% model.
%
% Can assess the log-likelihood of a given parameter combination (when
% provided both rewards and choices), or can simulate behavior for a given
% parameter combination (when given only rewards and parameters).
%
% Parameters:
%   params:  struct array with fields:
%       .alpha: learning rate (0.0 - 1.0)
%       .beta:  exploration ( > 0.0)
%       .V0:    initial value for expected reward
%       .split_learning (if TRUE, then use alphas below)
%       .alpha_win
%       .alpha_loss
%   rewards: (num_choices x T) matrix, where num_choices is number of
%            choices and T is number of trials, where rewards(k, t) is the
%            reward gained for choosing option k at time t.
%   choices: (1 x T) vector, where choices(t) is the selected choice at
%            timepoint t.
%
% Return Values:
%   expected_reward:(num_choices x T) matrix, where expected_reward(k, t)
%                   is the expected value for choice k at time t.
%   sim_choices:    (1 x T) vector, where sim_choices(t) is the selecetd
%                   choice at timepoint t, when simulating. NaN when not
%                   simulating (when choices are provided)
%   P:              A (num_choices x T) matrix, where P(k, t) is the
%                   probability of choosing choice k at time t.
%
% Written by: Samuel Taylor, Laureate Institute for Brain Research (2022)

function [model_output] = RW_model(params, rewards, choices, sim)


% params.V0 = 0;
%params.cr = 1;
%params.cl = 1;
% params.gamma = 0;

% if params.forgetting_bias
%     forgetting_bias = [params.psi_win params.psi_neutral params.psi_loss];
% end

% For a N-choice decision task.
N_CHOICES = 3;
% Total number of trials.
T = params.T;


% If choices are passed in, do not run as simulation, but instead use
% provided choices (typically used for fitting).
if sim == 0
    sim = false;
    sim_choices = NaN;
    % If choices are not passed in, run as a simulation instead, selecting
    % choices based on current value of expected reward at that timestep.
elseif sim == 1
    sim = true;
    sim_choices = zeros(1, T);
    sim_rewards = zeros(1, T);
end

if params.assoc
    associability = ones(N_CHOICES, T);
end

% Has dimensions for both time and number of choices to keep track of
% expceted reward over time.
expected_value = zeros(N_CHOICES, T);
info_bonus = zeros(N_CHOICES, T);
last_chosen = zeros(N_CHOICES, T);
%number_of_times_chosen = zeros(N_CHOICES, T);
%number_of_times_chosen(:,1) = 0.1;

% Represents the probability distribution of making a particular choice
% over time.
P = zeros(N_CHOICES, T);

% Set the initial values.
expected_value(:, 1) = params.V0;


action_probabilities = zeros(1, T);
prediction_error_sequence = zeros(1,T);


% For each trial (timestep)...
for t = 1:T
    
    if t <= 3
        actions(t) = choices(t);
        outcomes(t) = rewards(t);
        
        if sim == 1
            sim_choices(1,t) = choices(t);
            sim_rewards(1,t) = rewards(t);
        end
        
        if ~params.assoc
            % Copy previous values (to keep unchosen choices at
            % the same value from the previous timestep).
            expected_value(:, t + 1) = expected_value(:, t);
            last_chosen(:,t+1) = last_chosen(:,t);
            last_chosen(actions(t),t+1) = t;
            
            %number_of_times_chosen(:,t+1) = number_of_times_chosen(:,t);
            %number_of_times_chosen(actions(t),t+1) = number_of_times_chosen(actions(t),t)+1;
            
            % only update expected value
            if outcomes(t) == 1
                outcome_trans = 1*params.cr;
            elseif outcomes(t) == 2
                outcome_trans = 0; % transform neutral outcome to 0
            elseif outcomes(t) == 3
                outcome_trans = -1*params.cl; % transform loss outcome to -1
            end
            prediction_error = outcome_trans - expected_value(actions(t), t);
            
            % update mean of the chosen option
            if isfield(params, 'alpha_win')
                if outcomes(t) == 1
                    expected_value(actions(t), t+1) = expected_value(actions(t), t) + params.alpha_win*prediction_error;
                elseif outcomes(t) == 2
                    expected_value(actions(t), t+1) = expected_value(actions(t), t) + params.alpha_neutral*prediction_error;
                elseif outcomes(t) == 3
                    expected_value(actions(t), t+1) = expected_value(actions(t), t) + params.alpha_loss*prediction_error;
                end
            else
                expected_value(actions(t), t+1) = expected_value(actions(t), t) + params.alpha*prediction_error;
            end
            % forgetting for unchosen options
            unchose_opt_ls = find([1 2 3] ~= actions(t));
            if isfield(params, 'psi_win')
                if outcomes(t) == 1
                    expected_value(unchose_opt_ls, t+1) = (1-params.psi_win)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                elseif outcomes(t) == 2
                    expected_value(unchose_opt_ls, t+1) = (1-params.psi_neutral)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                elseif outcomes(t) == 3
                    expected_value(unchose_opt_ls, t+1) = (1-params.psi_loss)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                end
            elseif isfield(params, 'psi')
                expected_value(unchose_opt_ls, t+1) = (1-params.psi)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
            end
        elseif params.assoc
            % Copy previous values (to keep unchosen choices at
            % the same value from the previous timestep).
            expected_value(:, t + 1) = expected_value(:, t);
            last_chosen(:,t+1) = last_chosen(:,t);
            last_chosen(actions(t),t+1) = t;
            
            %number_of_times_chosen(:,t+1) = number_of_times_chosen(:,t);
            %number_of_times_chosen(actions(t),t+1) = number_of_times_chosen(actions(t),t)+1;
            
            % only update expected value
            if outcomes(t) == 1
                outcome_trans = 1*params.cr;
            elseif outcomes(t) == 2
                outcome_trans = 0; % transform neutral outcome to 0
            elseif outcomes(t) == 3
                outcome_trans = -1*params.cl; % transform loss outcome to -1
            end
            prediction_error = outcome_trans - expected_value(actions(t), t);
            
            % Copy previous associability values (to keep unchosen choices at
            % same value from previous timestep).
            associability(:, t + 1) = associability(:, t);
            
            % Update the associability estimate, as per below (equations 7 and
            % 7): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5760201/
            associability(actions(t), t + 1) = (1 - params.eta) * associability(actions(t), t) + params.eta * abs(prediction_error);
            
            % Keeps the associability values at a minimum of 0.05 and maximum of 1.
            associability(:, t + 1) = max(associability(:, t + 1), 0.05);
            associability(:, t + 1) = min(associability(:, t + 1), 1);
            
            
            % Update the expected reward of the selected choice.
            expected_value(actions(t), t + 1) = expected_value(actions(t), t) + params.alpha * associability(actions(t), t) * prediction_error;
            
            % forgetting
            if isfield(params, 'psi')
                unchose_opt_ls = find([1 2 3] ~= actions(t));
                expected_value(unchose_opt_ls, t+1) = (1-params.psi)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
            end
        end
    elseif t > 3
        % Transform the expected reward vector to a discrete probability
        % distribution using a softmax function (which includes `beta` as
        % an exploration parameter)
        
        if ~params.softmax
            info_bonus(:,t) = (t - last_chosen(:,t));
            %info_bonus(:,t) = sqrt(log(t)./number_of_times_chosen(:,t));
            
            expo = (exp(params.beta * (expected_value(:, t)+params.gamma*info_bonus(:,t)))+eps);
            for i = 1:length(expo)
                if isinf(expo(i))
                    expo(i) = realmax;
                end
            end
            P(:, t) = expo / (sum(expo));
        elseif params.softmax
            info_bonus(:,t) = (t - last_chosen(:,t));
            %info_bonus(:,t) = sqrt(log(t)./number_of_times_chosen(:,t));
            beta(t) = params.beta_0*((t/10).^params.c);
            P(:, t) = exp(beta(t) * (expected_value(:, t)+params.gamma*info_bonus(:,t))) / sum(exp(beta(t) * (expected_value(:, t)+params.gamma*info_bonus(:,t))));
        end
        
        % If not simulating, get choice selected at t.
        if sim == 0
            choice_at_t = choices(t);
            % If simulating, sample from P(:, t) instead
        elseif sim == 1
            %choice_at_t = randsample(1:N_CHOICES, 1, true, P(:, t));
            choice_at_t = find(rand < cumsum(P(:,t)),1);
            choice_ls(t) = choice_at_t;
            sim_choices(1,t) = choice_at_t;
            
        end
        
        % Store the probability of the chosen action being selected at this
        % timepoint.
        action_probabilities(t) = P(choice_at_t, t);
        
        
        if ~params.assoc
            % Copy previous values (to keep unchosen choices at
            % the same value from the previous timestep).
            expected_value(:, t + 1) = expected_value(:, t);
            last_chosen(:,t+1) = last_chosen(:,t);
            last_chosen(choice_at_t,t+1) = t;
            
            %number_of_times_chosen(:,t+1) = number_of_times_chosen(:,t);
            %number_of_times_chosen(choice_at_t,t+1) = number_of_times_chosen(choice_at_t,t)+1;
            
            if sim == 0
                if rewards(t) == 1
                    reward_trans = 1*params.cr;
                elseif rewards(t) == 2
                    reward_trans = 0; % transform neutral outcome to 0
                elseif rewards(t) == 3
                    reward_trans = -1*params.cl; % transform loss outcome to -1
                end
                prediction_error = reward_trans - expected_value(choice_at_t, t);
                prediction_error_sequence(t) = prediction_error;
                
                
                % update mean of the chosen option
                if isfield(params, 'alpha_win')
                    if rewards(t) == 1
                        expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha_win*prediction_error;
                    elseif rewards(t) == 2
                        expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha_loss*prediction_error;
                    elseif rewards(t) == 3
                        expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha_neutral*prediction_error;
                    end
                else
                    expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha*prediction_error;
                end
                % forgetting for unchosen options
                unchose_opt_ls = find([1 2 3] ~= choice_at_t);
                if isfield(params, 'psi_win')
                    if rewards(t) == 1
                        expected_value(unchose_opt_ls, t+1) = (1-params.psi_win)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                    elseif rewards(t) == 2
                        expected_value(unchose_opt_ls, t+1) = (1-params.psi_neutral)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                    elseif rewards(t) == 3
                        expected_value(unchose_opt_ls, t+1) = (1-params.psi_loss)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                    end
                elseif isfield(params, 'psi')
                    expected_value(unchose_opt_ls, t+1) = (1-params.psi)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                end
                
                
            elseif sim == 1
                outcome = find(rand < cumsum(params.BlockProbs(:, choice_at_t)), 1); % 1: win, 2: neutral, 3: loss
                sim_rewards(1, t) =  outcome;
                if sim_rewards(1, t) == 1
                    reward_trans = 1*params.cr;
                elseif sim_rewards(1, t) == 2
                    reward_trans = 0; % transform neutral value to 0
                elseif sim_rewards(1, t) == 3
                    reward_trans = -1*params.cl; % transform loss value to -1
                end
                prediction_error = reward_trans - expected_value(choice_at_t, t);
                prediction_error_sequence(t) = prediction_error;
                
                % update mean of the chosen option
                if isfield(params, 'alpha_win')
                    if rewards(t) == 1
                        expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha_win*prediction_error;
                    elseif rewards(t) == 2
                        expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha_neutral*prediction_error;
                    elseif rewards(t) == 3
                        expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha_loss*prediction_error;
                    end
                else
                    expected_value(choice_at_t, t+1) = expected_value(choice_at_t, t) + params.alpha*prediction_error;
                end
                
                % forgetting for unchosen options
                unchose_opt_ls = find([1 2 3] ~= choice_at_t);
                
                if isfield(params, 'psi_win')
                    if rewards(t) == 1
                        expected_value(unchose_opt_ls, t+1) = (1-params.psi_win)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                    elseif rewards(t) == 2
                        expected_value(unchose_opt_ls, t+1) = (1-params.psi_neutral)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                    elseif rewards(t) == 3
                        expected_value(unchose_opt_ls, t+1) = (1-params.psi_loss)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                    end
                elseif isfield(params, 'psi')
                    expected_value(unchose_opt_ls, t+1) = (1-params.psi)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
                end
                
            end
            
            
            %%%%% ASSOCIABILITY PART (IF USING IT IN THE MODEL) %%%%%
        elseif params.assoc
            % Copy previous values (to keep unchosen choices at
            % the same value from the previous timestep).
            expected_value(:, t + 1) = expected_value(:, t);
            last_chosen(:,t+1) = last_chosen(:,t);
            last_chosen(choice_at_t,t+1) = t;
            
            %number_of_times_chosen(:,t+1) = number_of_times_chosen(:,t);
            %number_of_times_chosen(choice_at_t,t+1) = number_of_times_chosen(choice_at_t,t)+1;
            
            if sim == 0
                if rewards(t) == 1
                    reward_trans = 1*params.cr;
                elseif rewards(t) == 2
                    reward_trans = 0; % transform neutral outcome to 0
                elseif rewards(t) == 3
                    reward_trans = -1*params.cl; % transform loss outcome to -1
                end
                prediction_error = reward_trans - expected_value(choice_at_t, t);
                prediction_error_sequence(t) = prediction_error;
                
                
            elseif sim == 1
                outcome = find(rand < cumsum(params.BlockProbs(:, choice_at_t)), 1); % 1: win, 2: neutral, 3: loss
                sim_rewards(1, t) =  outcome;
                
                if sim_rewards(1, t) == 1
                    reward_trans = 1*params.cr;
                elseif sim_rewards(1, t) == 2
                    reward_trans = 0; % transform neutral value to 0
                elseif sim_rewards(1, t) == 3
                    reward_trans = -1*params.cl; % transform loss value to -1
                end
                prediction_error = reward_trans - expected_value(choice_at_t, t);
                prediction_error_sequence(t) = prediction_error;
            end
            
            % Copy previous associability values (to keep unchosen choices at
            % same value from previous timestep).
            associability(:, t + 1) = associability(:, t);
            
            % Update the associability estimate, as per below (equations 7 and
            % 7): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5760201/
            associability(choice_at_t, t + 1) = (1 - params.eta) * associability(choice_at_t, t) + params.eta * abs(prediction_error);
            
            % Keeps the associability values at a minimum of 0.05 and maximum of 1.
            associability(:, t + 1) = max(associability(:, t + 1), 0.05);
            associability(:, t + 1) = min(associability(:, t + 1), 1);
            % Copy previous expected reward values (to keep unchosen choices at
            % same value from previous timestep).
            expected_value(:, t + 1) = expected_value(:, t);
            
            % Update the expected reward of the selected choice.
            expected_value(choice_at_t, t + 1) = expected_value(choice_at_t, t) + params.alpha * associability(choice_at_t, t) * prediction_error;
            
            % forgetting
            if isfield(params, 'psi')
                unchose_opt_ls = find([1 2 3] ~= choice_at_t);
                expected_value(unchose_opt_ls, t+1) = (1-params.psi)*(expected_value(unchose_opt_ls, t)-expected_value(unchose_opt_ls, 1)) + expected_value(unchose_opt_ls, 1);
            end
            % Trims final value in associability matrix (is an extra value beyond
            % trials).
            associability = associability(:, 1:T);
        end
        
        
        % Trims final value in expected reward matrix (is an extra value beyond
        % trials).
        expected_value = expected_value(:, 1:T);
    end
    
    % Store model variables for returning.
    if sim == 0
        model_output.choices(t) = choices(t);
        model_output.rewards(t) = rewards(t);
        
    elseif sim == 1
        model_output.choices(t) = sim_choices(t);
        model_output.rewards(t) = sim_rewards(t);
    end
    
    if params.assoc
        model_output.associability = associability;
    end
    model_output.prediction_errors = prediction_error_sequence;
    model_output.P = P;
    model_output.action_probabilities = action_probabilities;
    model_output.expected_value = expected_value;
    model_output.info_bonus = info_bonus;
    %model_output.number_of_times_chosen = number_of_times_chosen;
    
    
end