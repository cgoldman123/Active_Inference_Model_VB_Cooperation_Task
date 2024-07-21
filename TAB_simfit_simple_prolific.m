        
function sim_results = TAB_simfit_simple_prolific(fit_results)
        fit_results.DCM = rmfield(fit_results.DCM, {'M','Ep','Cp','F'});
        rewards = fit_results.DCM.U{:}-1;
        choices = fit_results.DCM.Y{:}-1;
        num_blocks = fit_results.DCM.MDP.NB;
        params = fit_results.parameters;
        params.forgetting_split = fit_results.DCM.MDP.forgetting_split;
        params.learning_split = fit_results.DCM.MDP.learning_split;
        params.T = fit_results.DCM.MDP.T;
        
        %% SIMULATE BEHAVIOR
        fprintf("Simulating behavior");
        MDP_Block = cell(1, num_blocks);
        for block = 1:num_blocks
            params.BlockProbs = fit_results.DCM.MDP.BlockProbs(:,:,block);
            MDP_Block{block} = Simple_TAB_model(params, rewards(:,block), choices(:,block), 1);
        end
        simmed_choices = cell2mat(cellfun(@(c) c.choices', MDP_Block, 'UniformOutput', false));
        simmed_outcomes = cell2mat(cellfun(@(c) c.outcomes', MDP_Block, 'UniformOutput', false));
        simmed_DCM.MDP = fit_results.DCM.MDP;
        simmed_DCM.field = fit_results.DCM.field;
        simmed_DCM.Y = {simmed_choices + 1};
        simmed_DCM.U = {simmed_outcomes+1};
               
        
        
        %% FIT SIMMED BEHAVIOR
        fprintf("Fitting simulated behavior");
        DCM = TAB_inversion_simple(simmed_DCM); 
        % re-transform values and compare prior with posterior estimates
        %--------------------------------------------------------------------------
        field = fieldnames(DCM.M.pE);
        for i = 1:length(field)
            if strcmp(field{i},'eta_neu')
                prior.eta_neu = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta_neu = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'eta_win')
                prior.eta_win = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta_win = 1/(1+exp(-DCM.Ep.(field{i})));  
            elseif strcmp(field{i},'eta_loss')
                prior.eta_loss = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta_loss = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'eta')
                prior.eta = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'omega') 
                prior.omega = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.omega = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'omega_win')
                prior.omega_win = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.omega_win = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'omega_loss')
                prior.omega_loss = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.omega_loss = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'alpha')
                prior.alpha = exp(DCM.M.pE.(field{i}));
                mdp.alpha = exp(DCM.Ep.(field{i}));
            elseif strcmp(field{i},'cr')
                prior.cr = exp(DCM.M.pE.(field{i}));
                mdp.cr = exp(DCM.Ep.(field{i}));
            elseif strcmp(field{i},'cl')
                prior.cl = exp(DCM.M.pE.(field{i}));
                mdp.cl = exp(DCM.Ep.(field{i}));
            elseif strcmp(field{i},'p_a')
                prior.p_a = exp(DCM.M.pE.(field{i}));
                mdp.p_a = exp(DCM.Ep.(field{i}));
            end
        end


        all_MDPs = [];
        params = DCM.MDP;
        U_Block = DCM.U{:}-1;
        rewards = reshape(U_Block,params.T,params.NB)';

        Y_Block = DCM.Y{:}-1;
        choices = reshape(Y_Block,params.T,params.NB)';

        mdp.T = params.T;
        mdp.learning_split = params.learning_split; % 1 = separate wins/losses, 0 = not
        mdp.forgetting_split = params.forgetting_split; % 1 = separate wins/losses, 0 = not

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
           % mdp.force_choice = params.force_choice(block,:);
           % mdp.force_outcome = params.force_outcome(block,:);
            MDP_Block{block} = Simple_TAB_model(mdp, rewards(block,:), choices(block,:), 0);
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
        
        fieldsToRemove = {'T', 'forgetting_split', 'learning_split'};
        % Remove the specified fields from the struct mdp
        mdp = rmfield(mdp, fieldsToRemove);

        sim_results.prior = prior;
        sim_results.parameters = mdp;
        sim_results.param_names = DCM.field;
        sim_results.DCM = DCM;
        sim_results.simulations = MDP_Block;
        sim_results.avg_action_prob = avg_action_prob;
        sim_results.model_acc = model_acc;
        
        
        
end
        