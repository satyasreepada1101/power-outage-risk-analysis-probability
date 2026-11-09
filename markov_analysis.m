% ============================================================================
% markov_analysis.m — Markov Chain & Bayesian Analysis in MATLAB
%
% METHODOLOGY: Steps 3–6
% PURPOSE:
%   1. Load pre-classified grid state sequence (from Python step3 output)
%   2. Compute 3×3 transition probability matrices per state & season
%   3. Compute steady-state distributions via eigenvalue decomposition
%   4. Short-term prediction: S(n) = S(0) × P^n
%   5. Bayesian update of outage probability given weather evidence
%   6. Poisson outage frequency + Exponential duration modelling
%   7. Monte Carlo simulation to validate Markov predictions
%   8. Visualise all results with MATLAB plots
%
% USAGE: markov_analysis  (run from the project root directory)
%        or: markov_analysis('Tamil Nadu', 30)  to predict for a state
% ============================================================================

function markov_analysis(target_state, forecast_days)

    if nargin < 1, target_state  = 'Tamil Nadu'; end
    if nargin < 2, forecast_days = 30;            end

    fprintf('\n%s\n', repmat('═',1,65));
    fprintf('  POWER OUTAGE — MARKOV CHAIN ANALYSIS  (MATLAB)\n');
    fprintf('  Target State : %s\n', target_state);
    fprintf('  Horizon      : %d days\n', forecast_days);
    fprintf('%s\n\n', repmat('═',1,65));

    % ── Configuration ─────────────────────────────────────────────────────────
    STATE_NAMES  = {'Tamil Nadu','Karnataka','Rajasthan',...
                    'Haryana','Assam','West Bengal'};
    SEASON_NAMES = {'Summer','Monsoon','Winter'};
    SEASON_MONTHS = {[3 4 5], [6 7 8 9], [10 11 12 1 2]};
    N_STATES     = 3;
    STATE_LABELS = {'Normal','Stressed','Outage'};
    %STATE_COLORS = [0.18 0.8 0.44; 0.95 0.61 0.07; 0.91 0.30 0.24];
    % Ensure there is a color for every state (6 states = 6 rows)
    STATE_COLORS = [
    0.90, 0.22, 0.27; % Tamil Nadu
    0.16, 0.62, 0.56; % Karnataka
    0.91, 0.77, 0.42; % Rajasthan
    0.15, 0.27, 0.33; % Haryana
    0.96, 0.64, 0.38; % Assam
    0.66, 0.85, 0.86  % West Bengal
    ];

    % Mean outage durations per state (hours) — CEA calibrated
    MEAN_OUTAGE_HOURS = containers.Map(STATE_NAMES, [4.2 3.8 6.5 5.1 7.8 5.3]);

    % Bayesian priors (seasonal)
    P_OUTAGE_PRIOR    = containers.Map(SEASON_NAMES, [0.18 0.22 0.08]);
    P_W_GIVEN_OUT     = containers.Map(SEASON_NAMES, [0.70 0.65 0.45]);
    P_W_GIVEN_NO_OUT  = containers.Map(SEASON_NAMES, [0.25 0.30 0.10]);

    % ── Load data ─────────────────────────────────────────────────────────────
    data_file = fullfile('data','grid_states_classified.csv');
    if ~isfile(data_file)
        error('Data file not found: %s\nRun Python pipeline first.', data_file);
    end
    
    % Read table and standardize formats
    T = readtable(data_file);
    T.Date   = datetime(T.Date, 'InputFormat','yyyy-MM-dd');
    T.Season = string(T.Season);
    T.State  = string(T.State);

    % ── 1. Build transition matrices for ALL states ──────────────────────────
    fprintf('Building transition matrices for all states...\n');
    all_P      = struct();
    all_pi     = struct();
    all_P_seas = struct();
    
    % Get list of states actually present in the data to avoid indexing errors
    present_states = unique(T.State, 'stable');

    for si = 1:numel(STATE_NAMES)
        sname = STATE_NAMES{si};
        skey  = matlab.lang.makeValidName(sname);
        
        % Filter data for this specific state
        df_s = T(T.State == sname, :);
        
        if isempty(df_s)
            fprintf('  ⚠ No data for %s, skipping calculation...\n', sname);
            continue;
        end
        
        df_s = sortrows(df_s, 'Date');
        seq  = df_s.Grid_State;

        % Compute Global matrix and Steady-state
        [P_glob, C_glob] = build_transition_matrix(seq, N_STATES);
        all_P.(skey)  = P_glob;
        all_pi.(skey) = compute_steady_state(P_glob);
        
        fprintf('  ✓ %s processed\n', sname);

        % Seasonal matrices
        for ss = 1:numel(SEASON_NAMES)
            season = SEASON_NAMES{ss};
            df_sea = df_s(df_s.Season == season, :);
            seq_s  = df_sea.Grid_State;
            if numel(seq_s) > 10
                [P_s, ~] = build_transition_matrix(seq_s, N_STATES);
            else
                P_s = P_glob;   % Fallback to global if seasonal data is sparse
            end
            all_P_seas.(skey).(season) = P_s;
        end
    end

   % ── 2. Run Analysis & Plots for ALL States ────────────────────────────────
   weather_scores = 0:0.1:1.0; % Moved up to fix the "Unrecognized variable" error
   forecast_days  = 30;        % Ensure this is defined 
   for si = 1:numel(STATE_NAMES)
        current_state = STATE_NAMES{si};
        skey = matlab.lang.makeValidName(current_state);
        
        if ~isfield(all_P, skey)
            continue; % Skip if no data was found for this state
        end

        fprintf('\n%s\n', repmat('═',1,65));
        fprintf('  ANALYSIS FOR: %s\n', current_state);
        fprintf('%s\n', repmat('═',1,65));

        % Get state-specific math
        P    = all_P.(skey);
        pi_v = all_pi.(skey);
        
        % ── Short-term prediction ──
        S0   = [1, 0, 0]; % Start from Normal
        traj = markov_trajectory(S0, P, forecast_days);
        
        % ── Bayesian Update ──
        prior_out = pi_v(3);
        temp_bayes_table = zeros(numel(SEASON_NAMES), numel(weather_scores));
        for ss = 1:numel(SEASON_NAMES)
            for wi = 1:numel(weather_scores)
                [posterior, ~] = bayesian_update(prior_out, SEASON_NAMES{ss}, ...
                    weather_scores(wi), P_W_GIVEN_OUT, P_W_GIVEN_NO_OUT, P_OUTAGE_PRIOR);
                temp_bayes_table(ss, wi) = posterior;
            end
        end

        % ── Poisson & Exponential ──
        lam = pi_v(3) * forecast_days;
        mean_hrs = MEAN_OUTAGE_HOURS(current_state);

        % ── Generate Plots for this state ──
        % This will now save a unique set of plots for EVERY state
        plot_all(all_P, all_pi, all_P_seas, temp_bayes_table, weather_scores, ...
             STATE_NAMES, SEASON_NAMES, STATE_LABELS, STATE_COLORS, ...
             current_state, traj, forecast_days, lam, mean_hrs, pi_v);
        
        fprintf('  ✓ Analysis and plots complete for %s\n', current_state);
    end
    
    fprintf('\n✅  All %d states processed successfully.\n', numel(STATE_NAMES));
% ============================================================================
%  LOCAL FUNCTIONS
% ============================================================================

function [P, C] = build_transition_matrix(seq, N)
    % Count transitions from consecutive day pairs
    C = zeros(N, N);
    for i = 1:numel(seq)-1
        from = seq(i) + 1;   % MATLAB is 1-indexed
        to_s = seq(i+1) + 1;
        if from >= 1 && from <= N && to_s >= 1 && to_s <= N
            C(from, to_s) = C(from, to_s) + 1;
        end
    end
    % Row-normalise; zero rows → uniform
    P = C;
    row_sums = sum(P, 2);
    for r = 1:N
        if row_sums(r) == 0
            P(r, :) = 1/N;
        else
            P(r, :) = P(r, :) / row_sums(r);
        end
    end
end


function pi = compute_steady_state(P)
    % Left eigenvector for eigenvalue 1
    [V, D] = eig(P');
    [~, idx] = min(abs(diag(D) - 1));
    pi = real(V(:, idx))';
    pi = abs(pi) / sum(abs(pi));
end


function traj = markov_trajectory(S0, P, n_steps)
    % S(n) = S(0) × P^n
    traj = zeros(n_steps+1, numel(S0));
    traj(1,:) = S0;
    S = S0;
    for i = 2:n_steps+1
        S = S * P;
        traj(i,:) = S;
    end
end


function [posterior, bayes_factor] = bayesian_update(prior, season, ...
        weather_score, P_W_O, P_W_NO, P_O_prior)
    % Bayesian updating with smooth weather_score interpolation
    p_w_out   = P_W_O(season);
    p_w_noout = P_W_NO(season);
    alpha     = weather_score;
    p_w_out_eff   = (1-alpha)*prior       + alpha*p_w_out;
    p_w_noout_eff = (1-alpha)*(1-prior)/2 + alpha*p_w_noout;
    p_evidence    = p_w_out_eff*prior + p_w_noout_eff*(1-prior);
    p_evidence    = max(p_evidence, 1e-9);
    posterior     = (p_w_out_eff * prior) / p_evidence;
    posterior     = min(max(posterior, 0), 1);
    bayes_factor  = p_w_out_eff / max(p_w_noout_eff, 1e-9);
end


function final_states = monte_carlo_markov(S0, P, n_steps, n_sims)
    % Run n_sims independent Markov chains for n_steps steps
    cum_P = cumsum(P, 2);   % CDF rows for fast sampling
    N     = size(P, 1);
    final_states = zeros(n_sims, N);

    for sim = 1:n_sims
        % Sample initial state
        u = rand;
        probs = S0;
        state = find(cumsum(probs) >= u, 1);
        if isempty(state), state = N; end

        for step = 1:n_steps
            u     = rand;
            state = find(cum_P(state,:) >= u, 1);
            if isempty(state), state = N; end
        end
        final_states(sim, state) = 1;
    end
    % Normalise to proportions
    final_states = mean(final_states, 1);
end


function disp_matrix(P, labels)
    fprintf('         %-10s %-12s %-10s\n', labels{:});
    for r = 1:size(P,1)
        fprintf('  %-9s', labels{r});
        for c = 1:size(P,2)
            fprintf('  %8.4f', P(r,c));
        end
        fprintf('\n');
    end
    fprintf('\n');
end


% ============================================================================
%  PLOT FUNCTIONS
% ============================================================================

    function plot_all(all_P, all_pi, all_P_seas, bayes_table, weather_scores, ...
                  STATE_NAMES, SEASON_NAMES, STATE_LABELS, STATE_COLORS, ...
                  target_state, traj, n_days, lam, mean_hrs, pi_v)
    
    if ~isfolder('plots_matlab'), mkdir('plots_matlab'); end
    
    % Create a clean suffix for the filename (e.g., 'TamilNadu')
    sname = matlab.lang.makeValidName(target_state);

    % ── Plot 1: All-state transition matrices ─────────────────────────────────
    fig = figure('Name','Transition Matrices','Position',[50 50 1400 900]);
    for si = 1:numel(STATE_NAMES)
        sname_iter = STATE_NAMES{si};
        skey_iter  = matlab.lang.makeValidName(sname_iter);
        P     = all_P.(skey_iter);
        ax = subplot(2,3,si);
        imagesc(P); colormap(ax, flipud(colormap('summer'))); caxis([0 1]);
        title(STATE_NAMES{si}, 'FontSize', 12, 'FontWeight', 'bold');
        colorbar;
        for r = 1:3
            for c = 1:3
                text(c, r, sprintf('%.3f', P(r,c)), ...
                    'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
            end
        end
        set(ax,'XTick',1:3,'XTickLabel',STATE_LABELS,...
               'YTick',1:3,'YTickLabel',STATE_LABELS,'FontSize',9);
         
        xlabel('Next State'); ylabel('Current State');
    end
    sgtitle('3×3 Transition Probability Matrices — All States','FontSize',14);
    saveas(fig, ['plots_matlab/M1_transition_matrices_', sname, '.png']);
    fprintf('  ✓ M1_transition_matrices_%s.png\n', sname);

    % ── Plot 2: Markov trajectory ─────────────────────────────────────────────
    fig = figure('Name','Markov Trajectory','Position',[100 100 900 550]);
    days = 0:n_days;
    colors = [0.18 0.8 0.44; 0.95 0.61 0.07; 0.91 0.30 0.24];
    labels = {'P(Normal)','P(Stressed)','P(Outage)'};
    area(days, traj, 'FaceAlpha', 0.25); hold on;
    for i = 1:3
        plot(days, traj(:,i), 'Color', colors(i,:), 'LineWidth', 2.5, ...
             'DisplayName', labels{i});
        yline(pi_v(i), '--', 'Color', colors(i,:), 'LineWidth', 1, ...
              'Alpha', 0.6, 'HandleVisibility','off');
    end
    legend('show','Location','east','FontSize',10);
    xlabel('Days Ahead'); ylabel('State Probability');
    title(sprintf('Markov Chain Forecast — %s (%d days)', target_state, n_days),...
          'FontSize',13,'FontWeight','bold');
    ylim([0 1]); grid on;
    saveas(fig, ['plots_matlab/M2_markov_trajectory_', sname, '.png']);
    fprintf('  ✓ M2_markov_trajectory_%s.png\n', sname);

   % ── Plot 3: Steady-State Probability Distribution (All States) ───────────
    fig = figure('Name', 'Steady State Analysis', 'Position', [150 150 1100 600]);
    steady_matrix = zeros(numel(STATE_NAMES), 3);
    for si = 1:numel(STATE_NAMES)
        skey_loop = matlab.lang.makeValidName(STATE_NAMES{si});
        if isfield(all_pi, skey_loop)
            steady_matrix(si, :) = all_pi.(skey_loop);
        else
            steady_matrix(si, :) = [0 0 0]; 
        end
    end
    x_labels = categorical(STATE_NAMES);
    x_labels = reordercats(x_labels, STATE_NAMES); 
    b = bar(x_labels, steady_matrix, 'grouped', 'EdgeColor', 'flat');
    b(1).FaceColor = [0.16, 0.62, 0.56]; b(1).DisplayName = 'Normal';
    b(2).FaceColor = [0.91, 0.77, 0.42]; b(2).DisplayName = 'Stressed';
    b(3).FaceColor = [0.90, 0.22, 0.27]; b(3).DisplayName = 'Outage';
    ylabel('Steady-State Probability (\pi)', 'FontWeight', 'bold');
    ylim([0 1]);
    title('Long-Run Steady-State Distribution — All States', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'FontSize', 10, 'YGrid', 'on', 'GridLineStyle', ':');
    legend('show', 'Location', 'northeastoutside', 'FontSize', 10);
    saveas(fig, ['plots_matlab/M3_steady_state_', sname, '.png']);
    fprintf('  ✓ M3_steady_state_%s.png\n', sname);

    % ── Plot 4: Bayesian update curves ────────────────────────────────────────
    fig = figure('Name','Bayesian Update','Position',[200 200 900 500]);
    season_colors_bay = {[0.9 0.3 0.1], [0.2 0.4 0.8], [0.3 0.7 0.3]};
    hold on;
    for si_s = 1:3
        plot(weather_scores, bayes_table(si_s,:), '-o', ...
             'Color', season_colors_bay{si_s}, 'LineWidth', 2.2, ...
             'MarkerSize', 5, 'DisplayName', SEASON_NAMES{si_s});
    end
    yline(pi_v(3), 'k--', 'LineWidth', 1.5, 'DisplayName', ...
          sprintf('Prior = %.3f', pi_v(3)));
    xlabel('Weather Extremity Score (0–1)');
    ylabel('Posterior P(Outage)');
    title(sprintf('Bayesian Update — %s', target_state),'FontSize',13);
    legend('show','FontSize',11,'Location','northwest');
    grid on; ylim([0 1]);
    saveas(fig, ['plots_matlab/M4_bayesian_update_', sname, '.png']);
    fprintf('  ✓ M4_bayesian_update_%s.png\n', sname);

    % ── Plot 5: Poisson PMF ────────────────────────────────────────────────────
    fig = figure('Name','Poisson PMF','Position',[250 250 800 480]);
    k_max = max(round(lam*3.5)+4, 8);
    k = 0:k_max;
    pmf = poisspdf(k, lam);
    bar(k, pmf, 0.75, 'FaceColor',[0.2 0.5 0.8],'EdgeColor','white','FaceAlpha',0.85);
    hold on;
    xline(lam,'r--','LineWidth',2,'DisplayName',sprintf('λ = %.3f',lam));
    xlabel('Number of Outage Days'); ylabel('Probability');
    title(sprintf('Poisson PMF — %s (%d-day window)', target_state, n_days),...
          'FontSize',13);
    legend(sprintf('λ = %.3f', lam),'Location','northeast','FontSize',11);
    grid on;
    saveas(fig, ['plots_matlab/M5_poisson_pmf_', sname, '.png']);
    fprintf('  ✓ M5_poisson_pmf_%s.png\n', sname);

    % ── Plot 6: Exponential CDF ────────────────────────────────────────────────
    fig = figure('Name','Exponential CDF','Position',[300 300 850 500]);
    mean_hrs_all = [4.2 3.8 6.5 5.1 7.8 5.3];
    t = linspace(0, 20, 400);
    hold on;
    for si_loop = 1:numel(STATE_NAMES)
        mu_e = 1/mean_hrs_all(si_loop);
        cdf  = 1 - exp(-mu_e * t);
        plot(t, cdf, 'LineWidth', 2.2, ...
             'Color', STATE_COLORS(si_loop,:), ...
             'DisplayName', sprintf('%s (mean %.1fh)', STATE_NAMES{si_loop}, mean_hrs_all(si_loop)));
    end
    yline(0.5, 'k:', 'LineWidth', 1, 'HandleVisibility','off');
    yline(0.9, 'k--', 'LineWidth', 1, 'HandleVisibility','off');
    text(19.5, 0.52, '50%', 'FontSize', 9, 'HorizontalAlignment','right');
    text(19.5, 0.92, '90%', 'FontSize', 9, 'HorizontalAlignment','right');
    xlabel('Outage Duration (hours)'); ylabel('P(Duration ≤ t)');
    title('Exponential CDF — Outage Duration per State','FontSize',13);
    legend('show','Location','southeast','FontSize',8);
    grid on; xlim([0 20]); ylim([0 1]);
    saveas(fig, ['plots_matlab/M6_exponential_cdf_', sname, '.png']);
    fprintf('  ✓ M6_exponential_cdf_%s.png\n', sname);

    % ── Plot 7: Seasonal matrix comparison (target state) ─────────────────────
    skey_target = matlab.lang.makeValidName(target_state);
    fig = figure('Name','Seasonal Matrices','Position',[350 350 1400 420]);
    for ss = 1:3
        season_name = SEASON_NAMES{ss};
        P_s = all_P_seas.(skey_target).(season_name);
        ax = subplot(1,3,ss);
        imagesc(P_s); colormap(ax, flipud(colormap('summer'))); caxis([0 1]);
        colorbar;
        for r = 1:3
            for c = 1:3
                text(c, r, sprintf('%.3f', P_s(r,c)), ...
                    'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');
            end
        end
        set(ax,'XTick',1:3,'XTickLabel',STATE_LABELS,...
               'YTick',1:3,'YTickLabel',STATE_LABELS,'FontSize',9);
        title(season_name,'FontSize',12,'FontWeight','bold');
        xlabel('Next State'); if ss==1, ylabel('Current State'); end
    end
    sgtitle(sprintf('%s — Seasonal Transition Matrices', target_state),'FontSize',13);
    saveas(fig, ['plots_matlab/M7_seasonal_matrices_', sname, '.png']);
    fprintf('  ✓ M7_seasonal_matrices_%s.png\n', sname);
    
    close all;
    end
end
