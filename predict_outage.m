% ============================================================================
% predict_outage.m — Interactive Power Outage Prediction Engine (MATLAB)
%
% PURPOSE:
%   Given a state, date, and optional weather inputs, produces a complete
%   outage risk assessment including:
%     • Seasonal Markov transition matrix selection
%     • S(n) = S(0) × P^n prediction trajectory
%     • Bayesian-updated outage probability
%     • Poisson expected outage count
%     • Exponential duration estimate
%     • A formatted prediction report + visualisation
%
% USAGE:
%   predict_outage('Rajasthan', '2024-05-20')
%   predict_outage('Assam',     '2024-07-10', 45, 120, 15, 30)
%   predict_outage('Haryana',   '2024-12-01')
%
% ARGUMENTS:
%   state_name  — string, one of the 6 target states
%   date_str    — 'YYYY-MM-DD'
%   temp_c      — max temperature in °C         (optional, seasonal default)
%   rain_mm     — precipitation in mm/day       (optional, default 2 or 30)
%   wind_ms     — wind speed in m/s             (optional, default 5 or 8)
%   horizon     — forecast horizon in days      (optional, default 30)
% ============================================================================


% ============================================================================
% predict_outage.m — Interactive Power Outage Prediction Engine (MATLAB)
%
% PURPOSE:
%   Given a state, date, and optional weather inputs, produces a complete
%   outage risk assessment including:
%     • Seasonal Markov transition matrix selection
%     • S(n) = S(0) × P^n prediction trajectory
%     • Bayesian-updated outage probability
%     • Poisson expected outage count
%     • Exponential duration estimate
%     • A formatted prediction report + visualisation
%
% USAGE:
%   predict_outage('Rajasthan', '2024-05-20')
%   predict_outage('Assam',     '2024-07-10', 45, 120, 15, 30)
%   predict_outage('Haryana',   '2024-12-01')
%
% ARGUMENTS:
%   state_name  — string, one of the 6 target states
%   date_str    — 'YYYY-MM-DD'
%   temp_c      — max temperature in °C         (optional, seasonal default)
%   rain_mm     — precipitation in mm/day       (optional, default 2 or 30)
%   wind_ms     — wind speed in m/s             (optional, default 5 or 8)
%   horizon     — forecast horizon in days      (optional, default 30)
% ============================================================================


function result = predict_outage(state_name, date_str, temp_c, rain_mm, wind_ms, horizon)

    % ── Defaults ──────────────────────────────────────────────────────────────
    if nargin < 1, state_name = 'Tamil Nadu'; end
    % --- Updated Default Date Logic ---

    if nargin < 2 
        % This gets the current system date and formats it as YYYY-MM-DD
        date_str = datestr(datetime('now'), 'yyyy-mm-dd'); 
    end
    if nargin < 6, horizon    = 30;           end

    % ── Config ────────────────────────────────────────────────────────────────
    STATE_NAMES  = {'Tamil Nadu','Karnataka','Rajasthan',...
                    'Haryana','Assam','West Bengal'};
    
    % --- FIXED SEASON MONTHS ---
    % April (4) and June (6) are now included in Summer and Monsoon respectively
    SEASON_MONTHS = struct(...
        'Summer',  {[3, 4, 5, 6]}, ...
        'Monsoon', {[7, 8, 9, 10]}, ...
        'Winter',  {[11, 12, 1, 2]} ...
    );
    SEASON_LIST  = {'Summer','Monsoon','Winter'};

    % State-specific temperature thresholds [stressed, outage]
    TEMP_THR = struct(...
        'Tamil_Nadu',  [35 40], 'Karnataka',   [33 38], ...
        'Rajasthan',   [38 44], 'Haryana',     [37 42], ...
        'Assam',       [32 36], 'West_Bengal', [34 39] ...
    );
    MEAN_OUTAGE_HOURS = containers.Map(STATE_NAMES, [4.2 3.8 6.5 5.1 7.8 5.3]);
    P_W_GIVEN_OUT     = struct('Summer',0.70,'Monsoon',0.65,'Winter',0.45);
    P_W_GIVEN_NO_OUT  = struct('Summer',0.25,'Monsoon',0.30,'Winter',0.10);

    % Seasonal weather defaults
    temp_defaults = struct(...
        'Tamil_Nadu',  struct('Summer',38,'Monsoon',32,'Winter',27), ...
        'Karnataka',   struct('Summer',36,'Monsoon',29,'Winter',24), ...
        'Rajasthan',   struct('Summer',42,'Monsoon',34,'Winter',22), ...
        'Haryana',     struct('Summer',41,'Monsoon',33,'Winter',20), ...
        'Assam',       struct('Summer',34,'Monsoon',31,'Winter',18), ...
        'West_Bengal', struct('Summer',37,'Monsoon',32,'Winter',22) ...
    );
    rain_defaults = struct('Summer',2,'Monsoon',30,'Winter',2);

    % ── Validate state ────────────────────────────────────────────────────────
    if ~ismember(state_name, STATE_NAMES)
        error('Unknown state ''%s''. Valid: %s', state_name, strjoin(STATE_NAMES,', '));
    end
    
    % Fixed State Key Handling
    skey = matlab.lang.makeValidName(state_name);
    if ~isfield(temp_defaults, skey)
        skey = strrep(state_name, ' ', '_'); 
    end

    % ── Parse date + season (FIXED) ───────────────────────────────────────────
    date_obj = datetime(date_str,'InputFormat','yyyy-MM-dd');
    % Use datevec to ensure month extraction works without toolboxes
    dv = datevec(date_obj); 
    month_num = dv(2);

    season = '';
    for i = 1:numel(SEASON_LIST)
        s_name = SEASON_LIST{i};
        if ismember(month_num, SEASON_MONTHS.(s_name))
            season = s_name;
            break;
        end
    end
    if isempty(season), season = 'Winter'; end

    % ── Fill weather defaults if not provided ─────────────────────────────────
    if nargin < 3 || isempty(temp_c)
        temp_c = temp_defaults.(skey).(season);
    end
    if nargin < 4 || isempty(rain_mm)
        rain_mm = rain_defaults.(season);
    end
    if nargin < 5 || isempty(wind_ms)
        wind_ms = 8.0 * (strcmp(season,'Summer')) + 5.0 * (~strcmp(season,'Summer'));
    end

    % ── Load Markov model ─────────────────────────────────────────────────────
    P_global  = load_or_build_matrix(state_name, 'global');
    P_season  = load_or_build_matrix(state_name, season);
    pi_steady = compute_steady_state(P_season);

    % ── Initial state: last recorded ─────────────────────────────────────────
    last_state = get_last_state(state_name);
    S0    = zeros(1, 3);
    S0(last_state + 1) = 1;

    % ── Markov trajectory ─────────────────────────────────────────────────────
    traj = markov_trajectory(S0, P_season, horizon);

    % ── Weather extremity score ───────────────────────────────────────────────
    thr     = TEMP_THR.(skey);
    t_score = max(0, min(1, (temp_c - thr(1)) / (thr(2) - thr(1) + 1e-6)));
    r_score = min(1, rain_mm / 50);
    w_score = min(1, wind_ms / 12);
    weather_score = max([t_score r_score w_score]);

    % ── Bayesian update ───────────────────────────────────────────────────────
    prior    = pi_steady(3);
    p_w_out  = P_W_GIVEN_OUT.(season);
    p_w_no   = P_W_GIVEN_NO_OUT.(season);
    alpha    = weather_score;
    p_w_eff  = (1-alpha)*prior       + alpha*p_w_out;
    p_wno_eff= (1-alpha)*(1-prior)/2 + alpha*p_w_no;
    p_evid   = max(p_w_eff*prior + p_wno_eff*(1-prior), 1e-9);
    posterior= min(max((p_w_eff * prior) / p_evid, 0), 1);
    bayes_factor = p_w_eff / max(p_wno_eff, 1e-9);

    % ── Blend Markov Day+1 with Bayesian posterior ─────────────────────────────
    blended_outage = 0.5 * traj(2,3) + 0.5 * posterior;

    % ── Poisson ───────────────────────────────────────────────────────────────
    lam          = blended_outage * horizon;
    p_at_least_1 = 1 - poisspdf(0, lam);

    % ── Exponential ──────────────────────────────────────────────────────────
    mean_hrs = MEAN_OUTAGE_HOURS(state_name);
    mu_exp   = 1 / mean_hrs;
    median_h = log(2) / mu_exp;
    p_gt_4h  = exp(-mu_exp * 4);

    % ── Package result ────────────────────────────────────────────────────────
    result = struct();
    result.state          = state_name;
    result.date           = date_str;
    result.season         = season;
    result.horizon        = horizon;
    result.initial_state  = last_state;
    result.P_seasonal     = P_season;
    result.trajectory     = traj;
    result.weather_score  = weather_score;
    result.prior_outage   = prior;
    result.posterior_outage = posterior;
    result.bayes_factor   = bayes_factor;
    result.blended_outage = blended_outage;
    result.poisson_lambda = lam;
    result.prob_at_least_1 = p_at_least_1;
    result.mean_hrs        = mean_hrs;
    result.median_hrs      = median_h;
    result.prob_gt_4h      = p_gt_4h;

    % ── Print report ─────────────────────────────────────────────────────────
    print_report(result, traj, pi_steady, horizon);

    % ── Plot ─────────────────────────────────────────────────────────────────
    plot_prediction(result, traj, pi_steady, horizon, lam, mean_hrs, mu_exp);

end


% ============================================================================
%  LOCAL FUNCTIONS
% ============================================================================

function P = load_or_build_matrix(state_name, season_or_global)
    try
        csv_path = fullfile('data', 'grid_states_classified.csv');
        if ~isfile(csv_path)
            error('no data');
        end
        T    = readtable(csv_path);
        T.State  = string(T.State);
        T.Season = string(T.Season);
        T.Date   = datetime(T.Date,'InputFormat','yyyy-MM-dd');
        T        = sortrows(T,'Date');

        df_s = T(T.State == state_name, :);
        if strcmp(season_or_global, 'global')
            seq = df_s.Grid_State;
        else
            seq = df_s.Grid_State(df_s.Season == season_or_global);
        end

        if numel(seq) < 10
            seq = df_s.Grid_State;  % fallback to global
        end

        [P, ~] = build_trans(seq, 3);
    catch
        P = fallback_matrix(state_name, season_or_global);
    end
end


function [P, C] = build_trans(seq, N)
    C = zeros(N, N);
    for i = 1:numel(seq)-1
        f = seq(i)+1; t = seq(i+1)+1;
        if f>=1&&f<=N&&t>=1&&t<=N
            C(f,t) = C(f,t)+1;
        end
    end
    P = C;
    rs = sum(P,2);
    for r=1:N
        if rs(r)==0, P(r,:)=1/N; else, P(r,:)=P(r,:)/rs(r); end
    end
end


function P = fallback_matrix(state, season)
    base = containers.Map( ...
        {'Tamil Nadu','Karnataka','Rajasthan','Haryana','Assam','West Bengal'}, ...
        {[0.80 0.15 0.05; 0.20 0.65 0.15; 0.08 0.22 0.70], ...
         [0.82 0.13 0.05; 0.22 0.65 0.13; 0.10 0.20 0.70], ...
         [0.75 0.17 0.08; 0.18 0.60 0.22; 0.07 0.20 0.73], ...
         [0.77 0.16 0.07; 0.20 0.62 0.18; 0.08 0.18 0.74], ...
         [0.70 0.20 0.10; 0.15 0.58 0.27; 0.05 0.18 0.77], ...
         [0.78 0.16 0.06; 0.19 0.64 0.17; 0.09 0.19 0.72]} ...
    );
    P = base(state);
    if strcmp(season,'Summer')
        P(1,3) = P(1,3) * 1.3;
        P = normalize_rows(P);
    elseif strcmp(season,'Monsoon')
        P(2,3) = P(2,3) * 1.4;
        P = normalize_rows(P);
    end
end


function P = normalize_rows(P)
    rs = sum(P,2);
    for r=1:size(P,1)
        if rs(r)>0, P(r,:)=P(r,:)/rs(r); end
    end
end


function pi = compute_steady_state(P)
    [V, D] = eig(P');
    [~, idx] = min(abs(diag(D)-1));
    pi = real(V(:,idx))';
    pi = abs(pi)/sum(abs(pi));
end


function traj = markov_trajectory(S0, P, n_steps)
    traj = zeros(n_steps+1, 3);
    traj(1,:) = S0;
    S = S0;
    for i=2:n_steps+1
        S = S*P;
        traj(i,:) = S;
    end
end


function last_state = get_last_state(state_name)
    try
        T = readtable(fullfile('data','grid_states_classified.csv'));
        T.State = string(T.State);
        T.Date  = datetime(T.Date,'InputFormat','yyyy-MM-dd');
        df_s    = sortrows(T(T.State==state_name,:),'Date');
        last_state = df_s.Grid_State(end);
    catch
        last_state = 0;
    end
end


function print_report(r, traj, pi_v, horizon)
    sep = repmat('═',1,62);
    fprintf('\n%s\n', sep);
    fprintf('  POWER OUTAGE PREDICTION REPORT\n');
    fprintf('  State  : %s\n', r.state);
    fprintf('  Date   : %s  (%s)\n', r.date, r.season);
    fprintf('  Horizon: %d days\n', horizon);
    fprintf('%s\n', sep);
    smap = {'Normal','Stressed','Outage'};
    fprintf('\n  Current Grid Status : %s\n', smap{r.initial_state+1});
    fprintf('\n  ── Markov Chain Forecast ──\n');
    for day = [1 3 7 14 30]
        if day <= horizon
            fprintf('  P(Outage) Day+%-2d  : %.2f%%\n', day, traj(day+1,3)*100);
        end
    end
    fprintf('  Long-run P(Outage) : %.2f%%  (steady-state)\n', pi_v(3)*100);
    fprintf('\n  ── Bayesian Update ──\n');
    fprintf('  Weather score      : %.2f\n', r.weather_score);
    fprintf('  Prior  P(Outage)   : %.2f%%\n', r.prior_outage*100);
    fprintf('  Posterior P(Outage): %.2f%%  ', r.posterior_outage*100);
    if r.posterior_outage > r.prior_outage
        fprintf('(↑ increased)\n');
    else
        fprintf('(↓ decreased)\n');
    end
    fprintf('  Bayes Factor       : %.3f\n', r.bayes_factor);
    fprintf('\n  ── Poisson Distribution ──\n');
    fprintf('  λ (expected outages): %.3f\n', r.poisson_lambda);
    fprintf('  P(≥1 outage)        : %.2f%%\n', r.prob_at_least_1*100);
    fprintf('\n  ── Exponential Duration ──\n');
    fprintf('  Mean duration       : %.1f hours\n', r.mean_hrs);
    fprintf('  Median duration     : %.2f hours\n', r.median_hrs);
    fprintf('  P(duration > 4h)    : %.2f%%\n', r.prob_gt_4h*100);
    fprintf('%s\n\n', sep);
end


function plot_prediction(r, traj, pi_v, horizon, lam, mean_hrs, mu_exp)
    if ~isfolder('plots_matlab'), mkdir('plots_matlab'); end
    fig = figure('Name','Prediction','Position',[50 50 1400 880]);

    colors = [0.18 0.8 0.44; 0.95 0.61 0.07; 0.91 0.30 0.24];
    days   = 0:horizon;
    labels = {'Normal','Stressed','Outage'};

    ax1 = subplot(2,3,1);
    hold(ax1,'on');
    for i=1:3
        fill(ax1,[days fliplr(days)],[traj(:,i)' zeros(1,horizon+1)], ...
             colors(i,:),'FaceAlpha',0.18,'EdgeColor','none');
        plot(ax1, days, traj(:,i), 'Color',colors(i,:),'LineWidth',2.2,...
             'DisplayName',labels{i});
        yline(pi_v(i),'--','Color',colors(i,:),'LineWidth',0.8,'Alpha',0.5,...
              'HandleVisibility','off');
    end
    xlabel(ax1,'Days Ahead'); ylabel(ax1,'Probability');
    title(ax1,sprintf('Markov Forecast — %s', r.state),'FontWeight','bold');
    legend(ax1,'show','Location','northeast','FontSize',8);
    ylim(ax1,[0 1]); grid(ax1,'on');

    ax2 = subplot(2,3,2);
    plot(ax2, days, traj(:,3)*100, 'r-','LineWidth',2.5);
    hold(ax2,'on');
    yline(pi_v(3)*100,'--','Color',[0.5 0.5 0.5],'LineWidth',1.5,...
          'DisplayName',sprintf('Steady-state (%.1f%%)',pi_v(3)*100));
    yline(r.posterior_outage*100,':','Color',[0.8 0.2 0.2],'LineWidth',1.5,...
          'DisplayName',sprintf('Bayesian posterior (%.1f%%)',r.posterior_outage*100));
    fill(ax2, [days fliplr(days)], [traj(:,3)'*100 zeros(1,horizon+1)],...
         [0.91 0.30 0.24], 'FaceAlpha',0.15,'EdgeColor','none');
    xlabel(ax2,'Days Ahead'); ylabel(ax2,'P(Outage) %');
    title(ax2,'Outage Probability Trajectory','FontWeight','bold');
    legend(ax2,'show','FontSize',8); grid(ax2,'on');

    ax3 = subplot(2,3,3);
    k_max = max(round(lam*3.5)+4, 8);
    k     = 0:k_max;
    pmf   = poisspdf(k, lam);
    bar(ax3, k, pmf, 0.75, 'FaceColor',[0.2 0.5 0.8],'EdgeColor','w','FaceAlpha',0.85);
    hold(ax3,'on');
    xline(lam,'r--','LineWidth',2);
    xlabel(ax3,'# Outage Days'); ylabel(ax3,'Probability');
    title(ax3,sprintf('Poisson PMF  λ=%.2f',lam),'FontWeight','bold');
    grid(ax3,'on');

    ax4 = subplot(2,3,4);
    t_h  = linspace(0, 24, 400);
    cdf  = 1 - exp(-mu_exp * t_h);
    plot(ax4, t_h, cdf, 'Color',[0.6 0.1 0.8],'LineWidth',2.5);
    hold(ax4,'on');
    yline(0.5,'k:','LineWidth',1); yline(0.9,'k--','LineWidth',1);
    text(23,0.52,'50%','FontSize',8,'HorizontalAlignment','right');
    text(23,0.92,'90%','FontSize',8,'HorizontalAlignment','right');
    xlabel(ax4,'Outage Duration (hours)'); ylabel(ax4,'P(D ≤ t)');
    title(ax4,sprintf('Exponential CDF  mean=%.1fh',mean_hrs),'FontWeight','bold');
    grid(ax4,'on'); xlim(ax4,[0 24]);

    ax5 = subplot(2,3,5);
    ws_range = linspace(0,1,50);
    posts_s  = arrayfun(@(ws) compute_posterior(r.prior_outage, r.season, ws), ws_range);
    plot(ax5, ws_range, posts_s*100, 'Color',[0.91 0.30 0.24],'LineWidth',2.5);
    hold(ax5,'on');
    scatter(ax5, r.weather_score, r.posterior_outage*100, 80,'k','filled',...
            'DisplayName',sprintf('Today (%.2f)', r.weather_score));
    yline(r.prior_outage*100, 'k--','LineWidth',1.2,...
          'DisplayName',sprintf('Prior (%.1f%%)',r.prior_outage*100));
    xlabel(ax5,'Weather Extremity Score'); ylabel(ax5,'Posterior P(Outage) %');
    title(ax5,sprintf('Bayesian Update — %s', r.season),'FontWeight','bold');
    legend(ax5,'show','FontSize',8); grid(ax5,'on');

    ax6 = subplot(2,3,6);
    axis(ax6,'off');
    risk_level = 'LOW';
    if r.posterior_outage >= 0.5
        risk_level = 'HIGH';
    elseif r.posterior_outage >= 0.30
        risk_level = 'MOD-HIGH';
    elseif r.posterior_outage >= 0.15
        risk_level = 'MODERATE';
    end
    annotation('textbox',[0.68 0.08 0.28 0.38],...
               'String',{sprintf('STATE: %s', r.state),...
                          sprintf('DATE:  %s', r.date),...
                          sprintf('SEASON: %s', r.season),...
                          ' ',...
                          sprintf('P(Outage) TODAY'),...
                          sprintf('  %.1f%%  →  %s', r.posterior_outage*100, risk_level),...
                          ' ',...
                          sprintf('Expected outages/%dd: %.2f', horizon, lam),...
                          sprintf('Median duration: %.1fh', r.median_hrs)},...
               'FontSize',9.5,'FitBoxToText','off','EdgeColor','k',...
               'BackgroundColor',[0.97 0.97 0.97],'FontWeight','normal');

    sgtitle(sprintf('Power Outage Prediction — %s  |  %s', r.state, r.date),...
            'FontSize',14,'FontWeight','bold');
    out_name = sprintf('plots_matlab/PRED_%s_%s.png', ...
                        strrep(r.state,' ','_'), strrep(r.date,'-',''));
    saveas(fig, out_name);
end


function posterior = compute_posterior(prior, season, ws)
    P_WO  = struct('Summer',0.70,'Monsoon',0.65,'Winter',0.45);
    P_WNO = struct('Summer',0.25,'Monsoon',0.30,'Winter',0.10);
    p_w_eff  = (1-ws)*prior       + ws*P_WO.(season);
    p_wno_eff= (1-ws)*(1-prior)/2 + ws*P_WNO.(season);
    p_evid   = max(p_w_eff*prior + p_wno_eff*(1-prior),1e-9);
    posterior= min(max((p_w_eff*prior)/p_evid,0),1);
end