function mf_results = coop_model_free(file)

        subdat = readtable(file);

        %==========================================================================
        %==========================================================================
        
        % if prolific file
        if contains(file, 'cooperation_task')
    
            TpB = 16;     % trials per block
            NB  = 30;     % number of blocks
            N   = TpB*NB; % trials per block * number of blocks

            first_game_trial = min(find(ismember(subdat.trial_type, 'MAIN_START'))) +2;
            clean_subdat = subdat(first_game_trial:end, :);

            trial_types = clean_subdat.trial_type(clean_subdat.event_type==3,:);
            location_code = zeros(NB, 3);
            force_choice = zeros(NB, 3);
            force_outcome = zeros(NB, 3);

            location_map = containers.Map({'g', 's', 'b'}, [2, 3, 4]);
            force_choice_map = containers.Map({'g', 's', 'b'}, [1, 2, 3]);
            force_outcome_map = containers.Map({'W', 'N', 'L'}, [1, 2, 3]);

            for i = 1:length(trial_types)
                underscore_indices = strfind(trial_types{i}, '_');
                letters = trial_types{i}(underscore_indices(1)+1:underscore_indices(1)+3);
                location_code(i, :) = arrayfun(@(c) location_map(c), letters);
                forced_letters = trial_types{i}(underscore_indices(2)+1:underscore_indices(2)+3);
                force_choice(i, :) = arrayfun(@(c) force_choice_map(c), forced_letters);
                forced_outcome_letters = trial_types{i}(underscore_indices(3)+1:underscore_indices(3)+3);
                force_outcome(i, :) = arrayfun(@(c) force_outcome_map(c), forced_outcome_letters);
            end

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
            sub.o = cell2mat(sub.o);

            for i = 1:NB
                for j = 1:TpB
                    if sub.u{16*(i-1)+j,1}(1) == 'l' %== "left"
                        sub.u{16*(i-1)+j,1} = location_code(i,1);
                    elseif sub.u{16*(i-1)+j,1}(1) == 'u' %== "up"
                        sub.u{16*(i-1)+j,1} = location_code(i,2);
                    elseif sub.u{16*(i-1)+j,1}(1) == 'r' %== "right"
                        sub.u{16*(i-1)+j,1} = location_code(i,3);
                    end
                end
            end

            sub.u = cell2mat(sub.u);

            o_all = [];
            u_all = [];

            for n = 1:NB
                o_all = [o_all sub.o((n*TpB-(TpB-1)):TpB*n,1)];
                u_all = [u_all sub.u((n*TpB-(TpB-1)):TpB*n,1)];
            end
        
        % if local file
        else
            first_game_trial = min(find(ismember(subdat.trial_type, 'MAIN_START'))) +2;
            clean_subdat = subdat(first_game_trial:end, :);


            %% 12. Set up model structure
            %==========================================================================
            %==========================================================================

            T = 16; % trials per block
            NB  = 22;     % number of blocks
            N   = T*NB; % trials per block * number of blocks


            trial_types_and_schedule = clean_subdat.trial_type(clean_subdat.event_code==4,:);
            split_cells = cellfun(@(x) strsplit(x, ' ', 'CollapseDelimiters', true), trial_types_and_schedule, 'UniformOutput', false);
            % Extract the first and second parts into separate cell arrays
            trial_types = cellfun(@(x) x{1}, split_cells, 'UniformOutput', false);
            full_schedule = cellfun(@(x) x{2}, split_cells, 'UniformOutput', false);
            split_cells = cellfun(@(x) strsplit(x, '-', 'CollapseDelimiters', true), full_schedule, 'UniformOutput', false);
            schedule.good_probabilities = cellfun(@(x) x{1}, split_cells, 'UniformOutput', false);
            schedule.safe_probabilities = cellfun(@(x) x{2}, split_cells, 'UniformOutput', false);
            schedule.bad_probabilities = cellfun(@(x) x{3}, split_cells, 'UniformOutput', false);


            location_code = zeros(NB, 3);
            force_choice = zeros(NB, 3);
            force_outcome = zeros(NB, 3);
            block_probs = zeros(3,3,NB);

            location_map = containers.Map({'g', 's', 'b'}, [2, 3, 4]);
            force_choice_map = containers.Map({'g', 's', 'b'}, [1, 2, 3]);
            force_outcome_map = containers.Map({'W', 'N', 'L'}, [1, 2, 3]);

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


            % parse observations and actions
            sub.o = clean_subdat.result(clean_subdat.event_code == 5);
            sub.u = clean_subdat.response(clean_subdat.event_code == 5);


            for i = 1:N
                if sub.o{i,1} == "pos"
                    sub.o{i,1} = 2;
                elseif sub.o{i,1} == "neut"
                    sub.o{i,1} = 3;
                elseif sub.o{i,1} == "neg"
                    sub.o{i,1} = 4;
                else
                    error("read in a result that wasn't pos, neut, or neg");
                end
            end
            sub.o = sub.o(1:NB*T,:);
            sub.o = cell2mat(sub.o);


            % lefty behavioral file 
            if any(contains(sub.u, 'd_')) && any(contains(sub.u, 'w_')) && any(contains(sub.u, 'a_'))
                for i = 1:NB
                    for j = 1:T
                        if sub.u{16*(i-1)+j,1}(1) == 'a' %== "left"
                            sub.u{16*(i-1)+j,1} = location_code(i,1);
                        elseif sub.u{16*(i-1)+j,1}(1) == 'w' %== "up"
                            sub.u{16*(i-1)+j,1} = location_code(i,2);
                        elseif sub.u{16*(i-1)+j,1}(1) == 'd' %== "right"
                            sub.u{16*(i-1)+j,1} = location_code(i,3);
                        else
                            error("read in an action that wasn't left, up, right");
                        end
                    end
                end
            % righty behavioral file
            else
                for i = 1:NB
                    for j = 1:T
                        if sub.u{16*(i-1)+j,1}(1) == 'l' %== "left"
                            sub.u{16*(i-1)+j,1} = location_code(i,1);
                        elseif sub.u{16*(i-1)+j,1}(1) == 'u' %== "up"
                            sub.u{16*(i-1)+j,1} = location_code(i,2);
                        elseif sub.u{16*(i-1)+j,1}(1) == 'r' %== "right"
                            sub.u{16*(i-1)+j,1} = location_code(i,3);
                        else
                            error("read in an action that wasn't left, up, right");
                        end
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
            
            
        end
        
        total_win=0; total_neut=0; total_lose=0;
        good_bandit_chosen=0; safe_bandit_chosen=0; bad_bandit_chosen=0;
        win_stay=[]; neut_stay=[]; lose_stay=[];
        for b = 1:NB
            block_choices = u_all(:,b);
            block_outcomes = o_all(:,b);
            for c = 4:length(block_choices)
                if block_choices(c) == 2
                    good_bandit_chosen = good_bandit_chosen+1;
                elseif block_choices(c) == 3
                    safe_bandit_chosen = safe_bandit_chosen+1;
                elseif block_choices(c) == 4
                    bad_bandit_chosen = bad_bandit_chosen+1;
                end
                
                % if previous win
                if block_outcomes(c-1) == 2
                    total_win = total_win+1;
                    % win_stay
                    if block_choices(c) == block_choices(c-1)
                        win_stay = [win_stay c+(b*16)];
                    end
                % previous neutral
                elseif block_outcomes(c-1) == 3
                    total_neut = total_neut+1;
                    % neutral stay
                    if block_choices(c) == block_choices(c-1)
                        neut_stay = [neut_stay c+(b*16)];
                    end
                % previous lose
                elseif block_outcomes(c-1) == 4
                    total_lose = total_lose+1;
                    % lose stay
                    if block_choices(c) == block_choices(c-1)
                        lose_stay = [lose_stay c+(b*16)];
                    end
                end
            end
        end
        mf_results.win_stay_prob = length(win_stay)/total_win;
        mf_results.neutral_stay_prob = length(neut_stay)/total_neut;
        mf_results.lose_stay_prob = length(lose_stay)/total_lose;
        mf_results.good_bandit_chosen = good_bandit_chosen;
        mf_results.safe_bandit_chosen = safe_bandit_chosen;
        mf_results.bad_bandit_chosen = bad_bandit_chosen;

        

   

end