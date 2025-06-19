        
function sim_results = TAB_simfit_simple_prolific(fit_results)
        params = fit_results.DCM.MDP;
        fit_results.DCM = rmfield(fit_results.DCM, {'M','Ep','Cp','F'});
        rewards = fit_results.DCM.U{:}-1;
        choices = fit_results.DCM.Y{:}-1;
        num_blocks = fit_results.DCM.MDP.NB;
        for i = 1:numel(fit_results.param_names)
            params.(fit_results.param_names{i}) = fit_results.parameters.(fit_results.param_names{i});
        end

        
        %% SIMULATE BEHAVIOR
        fprintf("Simulating behavior");
        MDP_Block = cell(1, num_blocks);
        for block = 1:num_blocks
            params.force_choice = fit_results.DCM.MDP.force_choice(block,:);
            params.force_outcome = fit_results.DCM.MDP.force_outcome(block,:);
            params.BlockProbs = fit_results.DCM.MDP.BlockProbs(:,:,block);
            MDP_Block{block} = Simple_TAB_model_v3(params, rewards(:,block), choices(:,block), 1);
        end
        simmed_choices = cell2mat(cellfun(@(c) c.choices', MDP_Block, 'UniformOutput', false));
        simmed_outcomes = cell2mat(cellfun(@(c) c.outcomes', MDP_Block, 'UniformOutput', false));
        simmed_DCM.MDP = fit_results.DCM.MDP;
        simmed_DCM.estimation_prior = fit_results.prior;

        simmed_DCM.field = fit_results.DCM.field;
        simmed_DCM.Y = {simmed_choices + 1};
        simmed_DCM.U = {simmed_outcomes+1};
               
        
        
        %% FIT SIMMED BEHAVIOR
        fprintf("Fitting simulated behavior");
        DCM = TAB_inversion_simple(simmed_DCM); 
        % re-transform values and compare prior with posterior estimates
        %--------------------------------------------------------------------------
        mdp.eta = params.eta;
        mdp.omega = params.omega;
        field = fieldnames(DCM.Ep);
        for i = 1:length(field)
            if ismember(field{i},{'alpha', 'beta', 'cs', 'p_a', 'cr', 'cl', 'beta_0','alpha_d'})
                prior.(field{i}) = exp(DCM.M.pE.(field{i}));
                mdp.(field{i}) = exp(DCM.Ep.(field{i}));
            elseif ismember(field{i}, {'eta_win', 'eta_loss', 'eta_neutral', 'eta'})
                prior.(field{i}) = exp(DCM.M.pE.(field{i}));
                mdp.(field{i}) = exp(DCM.Ep.(field{i}));
            elseif ismember(DCM.field{i},{'omega', 'omega_win', 'omega_loss','omega_neutral', 'opt','psi_0','psi_r'})
                prior.(field{i}) = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.(field{i}) = 1/(1+exp(-DCM.Ep.(field{i})));  
            else
                error('Need to transform parameter')
%                 prior.(field{i}) = (DCM.M.pE.(field{i}));
%                 mdp.(field{i}) = (DCM.Ep.(field{i}));
            end
        end


        all_MDPs = [];
        U_Block = DCM.U{:}-1;
        rewards = reshape(U_Block,params.T,params.NB)';

        Y_Block = DCM.Y{:}-1;
        choices = reshape(Y_Block,params.T,params.NB)';

        
        mdp.T = params.T;
        %if ~isfield(mdp, "opt")
        %    mdp.opt = params.opt;
        %end
        mdp.learning_split = params.learning_split; % 1 = separate wins/losses, 0 = not
        mdp.forgetting_split_matrix = params.forgetting_split_matrix; % 1 = separate wins/losses, 0 = not
        mdp.forgetting_split_row = params.forgetting_split_row; % 1 = separate wins/losses, 0 = not
        mdp.dynamic_forgetting = params.dynamic_forgetting;
        mdp.dynamic_decision_noise = params.dynamic_decision_noise;

        % %Simulate beliefs using fitted values
        for block=1:params.NB
            mdp.force_choice = fit_results.DCM.MDP.force_choice(block,:);
            mdp.force_outcome = fit_results.DCM.MDP.force_outcome(block,:);
            MDP_Block{block} = Simple_TAB_model_v3(mdp, rewards(block,:), choices(block,:), 0);
            % get avg action prob for free choices
            avg_act_probs(block) = sum(MDP_Block{block}.chosen_action_probabilities(4:end))/(mdp.T-3);
            

            for trial = 4:params.T
                chosen_act_prob = round(MDP_Block{block}.chosen_action_probabilities(trial),3);
                avg_act_prob = round(MDP_Block{block}.action_probabilities(:,trial),3);
                
                % if 3 options have the same maximum act prob
                if chosen_act_prob == max(avg_act_prob) & length(find(chosen_act_prob == avg_act_prob)) == 3
                    acc(block,trial-3) = 0.33;
                % if 2 options have the same maximum act prob
                elseif chosen_act_prob == max(avg_act_prob) & length(find(chosen_act_prob == avg_act_prob)) == 2
                    acc(block,trial-3) = 0.50;
                % if only 1 option has the same maximum act prob
                elseif chosen_act_prob == max(avg_act_prob) & length(find(chosen_act_prob == avg_act_prob)) == 1
                    acc(block,trial-3) = 1;
                else
                    acc(block, trial-3) = 0;
                end
            end
        end

        
        avg_action_prob = sum(avg_act_probs)/params.NB;
        model_acc = (sum(sum(acc,2))/(params.NB*(params.T-3)));
        
        for i=1:length(field)
            posterior_params.(field{i}) = mdp.(field{i});
        end
        sim_results.prior = prior;
        sim_results.parameters = posterior_params;
        sim_results.param_names = DCM.field;
        sim_results.DCM = DCM;
        sim_results.simulations = MDP_Block;
        sim_results.avg_action_prob = avg_action_prob;
        sim_results.model_acc = model_acc;
        
        
        
end
        