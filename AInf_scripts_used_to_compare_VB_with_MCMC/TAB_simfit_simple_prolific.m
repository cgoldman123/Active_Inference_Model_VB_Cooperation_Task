        
function sim_results = TAB_simfit_simple_prolific(fit_results)
        fit_results.DCM = rmfield(fit_results.DCM, {'M','Ep','Cp','F'});
        rewards = fit_results.DCM.U{:}-1;
        choices = fit_results.DCM.Y{:}-1;
        num_blocks = fit_results.DCM.MDP.NB;
        params = fit_results.parameters;
        params.forgetting_split_matrix = fit_results.DCM.MDP.forgetting_split_matrix;
        params.forgetting_split_row = fit_results.DCM.MDP.forgetting_split_row;
        params.learning_split = fit_results.DCM.MDP.learning_split;
        params.T = fit_results.DCM.MDP.T;
        
        %% SIMULATE BEHAVIOR
        fprintf("Simulating behavior");
        MDP_Block = cell(1, num_blocks);
        for block = 1:num_blocks
            params.force_choice = fit_results.DCM.MDP.force_choice(block,:);
            params.force_outcome = fit_results.DCM.MDP.force_outcome(block,:);
            params.BlockProbs = fit_results.DCM.MDP.BlockProbs(:,:,block);
            MDP_Block{block} = Simple_TAB_model_v2(params, rewards(:,block), choices(:,block), 1);
        end
        simmed_choices = cell2mat(cellfun(@(c) c.choices', MDP_Block, 'UniformOutput', false));
        simmed_outcomes = cell2mat(cellfun(@(c) c.outcomes', MDP_Block, 'UniformOutput', false));
        simmed_DCM.MDP = fit_results.DCM.MDP;

        simmed_DCM.field = fit_results.DCM.field;
        simmed_DCM.Y = {simmed_choices + 1};
        simmed_DCM.U = {simmed_outcomes+1};
               
        
        
        %% FIT SIMMED BEHAVIOR
        fprintf("Fitting simulated behavior");
        DCM = TAB_inversion_simple_untransformed(simmed_DCM); 
        % re-transform values and compare prior with posterior estimates
        %--------------------------------------------------------------------------
        field = fieldnames(DCM.M.pE);
        for i = 1:length(field)
            if ismember(field{i},{'alpha', 'beta', 'cs', 'p_a', 'cr', 'cl'})
                prior.(field{i}) = exp(DCM.M.pE.(field{i}));
                mdp.(field{i}) = exp(DCM.Ep.(field{i}));
            elseif ismember(DCM.field{i},{'eta_win', 'eta_loss', 'eta_neutral', 'eta', 'omega', 'omega_win', 'omega_loss','omega_neutral', 'opt'})
                prior.(field{i}) = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.(field{i}) = 1/(1+exp(-DCM.Ep.(field{i})));  
            else
                prior.(field{i}) = (DCM.M.pE.(field{i}));
                mdp.(field{i}) = (DCM.Ep.(field{i}));
            end
        end


        all_MDPs = [];
        params = DCM.MDP;
        U_Block = DCM.U{:}-1;
        rewards = reshape(U_Block,params.T,params.NB)';

        Y_Block = DCM.Y{:}-1;
        choices = reshape(Y_Block,params.T,params.NB)';

        mdp.T = params.T;
        mdp.forgetting_split_matrix = params.forgetting_split_matrix; % 1 = separate wins/losses, 0 = not
        mdp.forgetting_split_row = params.forgetting_split_row; % 1 = separate wins/losses, 0 = not
        mdp.learning_split = params.learning_split;


        %     %if splitting learning rates
        %          params.omega_win = 1;
        %          params.omega_loss = 1;
        % 
        %     %if splitting learning rates
        %          params.eta_win = .5;
        %          params.eta_loss = .5;
        %         
        % %Simulate beliefs using fitted values
        for block=1:params.NB
            mdp.force_choice = fit_results.DCM.MDP.force_choice(block,:);
            mdp.force_outcome = fit_results.DCM.MDP.force_outcome(block,:);
            MDP_Block{block} = Simple_TAB_model_v2(mdp, rewards(block,:), choices(block,:), 0);
            % get avg action prob for free choices
            avg_act_probs(block) = sum(MDP_Block{block}.chosen_action_probabilities(4:end))/(mdp.T-3);
            

            for trial = 4:params.T
                if MDP_Block{block}.chosen_action_probabilities(trial) == max(MDP_Block{block}.action_probabilities(:,trial))
                    acc(block,trial-3) = 1;
                else
                    acc(block,trial-3) = 0;
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
        