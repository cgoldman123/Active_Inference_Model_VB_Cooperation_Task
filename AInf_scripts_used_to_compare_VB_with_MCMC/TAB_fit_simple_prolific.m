function [fit_results,file] = TAB_fit_simple_prolific(subject,DCM)
    %% Add Subj Data (Parse the data files)
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    file_path = [root 'NPC/DataSink/StimTool_Online/WB_Cooperation_Task/'];
    
    
    has_practice_effects = false;
    % Manipulate Data
    directory = dir(file_path);
    % sort by date
    dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    % Sort the dates and get the sorted indices
    [~, sortedIndices] = sort(dates);
    % Use the sorted indices to sort the structure array
    sortedDirectory = directory(sortedIndices);

    index_array = find(arrayfun(@(n) contains(sortedDirectory(n).name, strcat('cooperation_task_',subject)),1:numel(sortedDirectory)));
    if length(index_array) > 1
        disp("WARNING, MULTIPLE BEHAVIORAL FILES FOUND FOR THIS ID. USING THE FIRST FULL ONE")
    end

    for k = 1:length(index_array)
        file_index = index_array(k);
        file = [file_path sortedDirectory(file_index).name];

        subdat = readtable(file);
        % if they got past "MAIN_START" but don't have the right number
        % of trials, indicate that they have practice effects and advance 
        % them to the next file
        if any(cellfun(@(x) isequal(x, 'MAIN_START'), subdat.trial_type))
            first_game_trial = min(find(ismember(subdat.trial_type, 'MAIN_START'))) +2;
            clean_subdat = subdat(first_game_trial:end, :);
            % make sure correct number of trials
            % note that event_type ==5 should always be 480 but sometimes
            % event_type ==4 will be 479
            if (length(clean_subdat.result(clean_subdat.event_type == 5)) ~= 480) || (length(clean_subdat.response(clean_subdat.event_type == 5)) ~= 480)
                has_practice_effects = true;
                continue;
            end
        else
            continue;
        end
        
        %% 12. Set up model structure
        %==========================================================================
        %==========================================================================

        T = DCM.MDP.T; % trials per block
        NB  = DCM.MDP.NB;     % number of blocks
        N   = T*NB; % trials per block * number of blocks


        trial_types = clean_subdat.trial_type(clean_subdat.event_type==3,:);
        location_code = zeros(NB, 3);
        force_choice = zeros(NB, 3);
        force_outcome = zeros(NB, 3);
        block_probs = zeros(3,3,NB);

        location_map = containers.Map({'g', 's', 'b'}, [2, 3, 4]);
        force_choice_map = containers.Map({'g', 's', 'b'}, [1, 2, 3]);
        force_outcome_map = containers.Map({'W', 'N', 'L'}, [1, 2, 3]);
%         schedule = readtable([root 'rsmith/lab-members/osanchez/wellbeing/cooperation/BV/task_schedule/prolific_30_block_schedule.xlsx']);
        schedule = readtable([root 'rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/task_schedule/prolific_30_block_schedule.xlsx']);

        for i = 1:length(trial_types)
            underscore_indices = strfind(trial_types{i}, '_');
            letters = trial_types{i}(underscore_indices(1)+1:underscore_indices(1)+3);
            location_code(i, :) = arrayfun(@(c) location_map(c), letters);
            forced_letters = trial_types{i}(underscore_indices(2)+1:underscore_indices(2)+3);
            force_choice(i, :) = arrayfun(@(c) force_choice_map(c), forced_letters);
            forced_outcome_letters = trial_types{i}(underscore_indices(3)+1:underscore_indices(3)+3);
            force_outcome(i, :) = arrayfun(@(c) force_outcome_map(c), forced_outcome_letters);
   
            block_probs(:,1,i) = str2double(strsplit(schedule.good_probabilities{i},'_'))';
            block_probs(:,2,i) = str2double(strsplit(schedule.safe_probabilities{i},'_'))';
            block_probs(:,3,i) = str2double(strsplit(schedule.bad_probabilities{i},'_'))';
        end
        
        
        
        %BlockProbs = load('BlockProbs_all.mat').BlockProbs_all;
        %force_choice = load('force_choice.mat').force_choice;
        %force_outcome = load('force_outcome.mat').force_outcome;
        %location_code = load('location_code.mat').location_code;

        DCM.MDP.force_choice = force_choice;
        DCM.MDP.force_outcome = force_outcome;
        DCM.MDP.BlockProbs = block_probs;
            %--------------------------------------------------------------------------





        % parse observations and actions
        sub.o = clean_subdat.result(clean_subdat.event_type == 5);
        sub.u = clean_subdat.response(clean_subdat.event_type == 5);
        

        for i = 1:N
            if sub.o{i,1} == "positive"
                sub.o{i,1} = 2;
            elseif sub.o{i,1} == "neutral"
                sub.o{i,1} = 3;
            elseif sub.o{i,1} == "negative"
                sub.o{i,1} = 4;
            end
        end
        sub.o = sub.o(1:NB*T,:);
        sub.o = cell2mat(sub.o);

        for i = 1:NB
            for j = 1:T
                if sub.u{16*(i-1)+j,1}(1) == 'l' %== "left"
                    sub.u{16*(i-1)+j,1} = location_code(i,1);
                elseif sub.u{16*(i-1)+j,1}(1) == 'u' %== "up"
                    sub.u{16*(i-1)+j,1} = location_code(i,2);
                elseif sub.u{16*(i-1)+j,1}(1) == 'r' %== "right"
                    sub.u{16*(i-1)+j,1} = location_code(i,3);
                end
            end
        end
        sub.u = sub.u(1:NB*T,:);
        sub.u = cell2mat(sub.u);

        o_all = [];
        u_all = [];

        for n = 1:NB
            o_all = [o_all sub.o((n*T-(T-1)):T*n,1)];
            u_all = [u_all sub.u((n*T-(T-1)):T*n,1)];
        end
        %% 6.2 Invert model and try to recover original parameters:
        %==========================================================================

        %--------------------------------------------------------------------------
        % This is the model inversion part. Model inversion is based on variational
        % Bayes. The basic idea is to maximise (negative) variational free energy
        % wrt to the free parameters (here: alpha and cr). This means maximising
        % the likelihood of the data under these parameters (i.e., maximise
        % accuracy) and at the same time penalising for strong deviations from the
        % priors over the parameters (i.e., minimise complexity), which prevents
        % overfitting.
        % 
        % You can specify the prior mean and variance of each parameter at the
        % beginning of the TAB_spm_dcm_mdp script.
        %--------------------------------------------------------------------------

      %  params.BlockProbs = BlockProbs;
        DCM.MDP.NB = NB;
        DCM.MDP.T = T;




        DCM.U      = {o_all};              % trial specification (stimuli)
        DCM.Y      = {u_all};              % responses (action)

        DCM        = TAB_inversion_simple_untransformed(DCM);   % Invert the model

        %% 6.3 Check deviation of prior and posterior means & posterior covariance:
        %==========================================================================

        %--------------------------------------------------------------------------
        % re-transform values and compare prior with posterior estimates
        %--------------------------------------------------------------------------
        mdp = DCM.MDP;
        field = fieldnames(DCM.M.pE);
        for i = 1:length(field)
            if ismember(field{i},{'alpha', 'beta', 'cs', 'p_a', 'cr', 'cl'})
                prior.(field{i}) = exp(DCM.M.pE.(field{i}));
                mdp.(field{i}) = exp(DCM.Ep.(field{i}));
            elseif ismember(field{i},{'eta_win', 'eta_loss', 'eta_neutral', 'eta', 'omega', 'omega_win', 'omega_loss','omega_neutral', 'opt'})
                prior.(field{i}) = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.(field{i}) = 1/(1+exp(-DCM.Ep.(field{i})));  
            else
                prior.(field{i}) = (DCM.M.pE.(field{i}));
                mdp.(field{i}) = (DCM.Ep.(field{i}));
            end
        end


        all_MDPs = [];

        U_Block = DCM.U{:}-1;
        rewards = reshape(U_Block,T, NB)';

        Y_Block = DCM.Y{:}-1;
        choices = reshape(Y_Block,T,NB)';


        for block=1:NB
            MDP_Block{block} = Simple_TAB_model_v2(mdp, rewards(block,:), choices(block,:), 0);
            % get avg action prob for free choices
            avg_act_probs(block) = sum(MDP_Block{block}.chosen_action_probabilities(4:end))/(mdp.T-3);

            for trial = 4:T
                if MDP_Block{block}.chosen_action_probabilities(trial) == max(MDP_Block{block}.action_probabilities(:,trial))
                    acc(block,trial-3) = 1;
                else
                    acc(block,trial-3) = 0;
                end
            end
        end

        
        avg_action_prob = sum(avg_act_probs)/NB;
        model_acc = (sum(sum(acc,2))/(NB*(T-3)));

        for i=1:length(field)
            posterior_params.(field{i}) = mdp.(field{i});
        end

        fit_results.file = {file};
        fit_results.num_blocks = NB;
        fit_results.prior = prior;
        fit_results.parameters = posterior_params;
        fit_results.param_names = DCM.field;
        fit_results.DCM = DCM;
        fit_results.MDP_block = MDP_Block;
        fit_results.avg_action_prob = avg_action_prob;
        fit_results.model_acc = model_acc;
        fit_results.has_practice_effects = has_practice_effects;

        fieldnames_DCM = fieldnames(DCM.MDP);
        for i = 1:length(fieldnames(DCM.MDP))
            if contains(fieldnames_DCM{i}, {'forgetting_split','learning_split',...
            'T', 'NB', 'force_choice','force_outcome', 'BlockProbs', 'forgetting_split_row','forgetting_split_matrix'})
            else
                if ~contains(fieldnames_DCM{i}, fit_results.param_names)
                    fit_results.fixed.(fieldnames_DCM{i}) = DCM.MDP.(fieldnames_DCM{i});
                end
            end
        end
      
        break;
        
        
    end
        

end