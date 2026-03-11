function structured_full_dimension_C_with_v1_plot()
%==========================================================================
% Compact public-release implementation of the full-dimensional corridor
% shortest-path experiment used as a reader-facing reference.
%
% We compare:
%   (1) Full SPO+ training in the ambient dimension d
%   (2) The discrete-path SDS learner included in this repository plus
%       reduced SPO+ training in the learned subspace
%
% Key property:
%   The prior set C is full-dimensional (affdim(C)=d), while the
%   decision-relevant dimension d_* is controlled by the corridor design.
%
% Outputs:
%   - Figure 1: log10(Test SPO risk) vs #labeled samples (mean +/- 90% CI)
%   - Figure 2: learned dim(W) vs #labeled samples (mean +/- 90% CI)
%   - Saved PNG figures and a MAT summary under results/
%
% Requires Optimization Toolbox for linprog (used in the SDS check).
%==========================================================================

clc; close all;

if exist('linprog','file') ~= 2
    error('This script requires Optimization Toolbox (linprog).');
end

% -------------------------- user controls ------------------------------
cfg.seed         = 20260311;
cfg.g            = 4;          % 4x4 grid
cfg.dstar_target = 6;          % desired d_* in {0,...,9}
cfg.p            = 5;          % context dimension
cfg.trainSizes   = [20 40 80 160 320];
cfg.nTest        = 2000;
cfg.nTrial       = 10;

% Corridor vs outside cost boxes (full-dimensional box in R^d)
cfg.lowBase  = 10;  cfg.radCorr = 1;    % corridor edges in [9,11]
cfg.highBase = 100; cfg.radOut  = 1;    % outside edges  in [99,101]

% Context signal & noise (kept small so costs stay inside the box)
cfg.signalStrength = 0.80;  % fraction of corridor radius used by contextual signal
cfg.noiseCorr      = 0.05;  % additive uniform noise on corridor edges
cfg.noiseOut       = 0.05;  % additive uniform noise on outside edges

% SPO+ training hyperparameters (simple Adam)
cfg.numEpochs = 40;
cfg.lr        = 0.05;

% SDS / feasibility tolerance
cfg.sdsTol    = 1e-9;

rng(cfg.seed, 'twister');

% -------------------------- build paths (oracle) -----------------------
[Z, ~, edge] = build_grid_paths(cfg.g);
[d, K] = size(Z);

% -------------------------- build corridor with target d_* -------------
[corrEdges, corrPathIdx, dstar_true] = find_corridor_custom_dstar(cfg.g, Z, edge, cfg.dstar_target);

% Full-dimensional box bounds
lbC = (cfg.highBase - cfg.radOut) * ones(d,1);
ubC = (cfg.highBase + cfg.radOut) * ones(d,1);
lbC(corrEdges) = cfg.lowBase - cfg.radCorr;
ubC(corrEdges) = cfg.lowBase + cfg.radCorr;

affdimC = sum(ubC > lbC + 1e-12);

% Baseline cost used by BOTH models (important in corridor construction):
% It encodes the coarse outside-is-expensive and corridor-is-cheap structure.
cBase = 0.5 * (lbC + ubC);

% True decision-difference subspace (for data generation only)
Wtrue  = Z(:,corrPathIdx) - Z(:,corrPathIdx(1));
Ustar  = orth(Wtrue);
rstar  = size(Ustar,2);

% Safe scaling for contextual signal so we mostly stay within bounds
edgeL1    = sum(abs(Ustar), 2);
maxL1     = max(edgeL1);
signalAmp = cfg.signalStrength * cfg.radCorr / max(maxL1, 1e-12);

% Fixed latent linear map from context to the d_* latent coordinates
Atrue = randn(rstar, cfg.p);

% -------------------------- sanity prints ------------------------------
fprintf('==== Corridor CLO (4x4) ====\n');
fprintf('d = %d edges, K = %d monotone paths.\n', d, K);
fprintf('affdim(C) = %d (should be 24).\n', affdimC);
fprintf('Target d_* = %d.  Achieved d_* = %d.\n', cfg.dstar_target, dstar_true);
fprintf('Corridor edges = %d / %d.\n', numel(corrEdges), d);
fprintf('Corridor paths = %d / %d.\n', numel(corrPathIdx), K);
fprintf('signalAmp = %.4f (keeps contextual signal within corridor bounds).\n', signalAmp);

% Hard domination check: any outside-path must be worse than any corridor-path for all c in C
verify_domination(Z, corrPathIdx, lbC, ubC);

% -------------------------- run experiment -----------------------------
nN = numel(cfg.trainSizes);
testRisk_full = zeros(cfg.nTrial, nN);
testRisk_red  = zeros(cfg.nTrial, nN);
dimW_learned  = zeros(cfg.nTrial, nN);

for ii = 1:nN
    nTrain = cfg.trainSizes(ii);
    fprintf('\n--- nTrain = %d ---\n', nTrain);

    for tr = 1:cfg.nTrial
        rng(cfg.seed + 100*ii + tr, 'twister');

        % Sample train/test data
        [Xtr, Ctr] = sample_contextual_costs(nTrain, cfg.p, cBase, Ustar, Atrue, signalAmp, ...
            cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corrEdges);

        [Xte, Cte] = sample_contextual_costs(cfg.nTest, cfg.p, cBase, Ustar, Atrue, signalAmp, ...
            cfg.noiseCorr, cfg.noiseOut, lbC, ubC, corrEdges);

        % (1) Full SPO+
        Bfull = train_spoplus_full(Xtr, Ctr, Z, cBase, cfg.numEpochs, cfg.lr);

        % (2) Repository SDS learner -> learned subspace -> reduced SPO+
        Ulearn = learn_sds_basis(Ctr, Z, lbC, ubC, cfg.sdsTol);
        Greduced = train_spoplus_reduced(Xtr, Ctr, Z, cBase, Ulearn, cfg.numEpochs, cfg.lr);

        % Evaluate SPO risk
        testRisk_full(tr, ii) = mean_spo_risk_full(Bfull, Xte, Cte, Z, cBase);
        testRisk_red(tr,  ii) = mean_spo_risk_reduced(Greduced, Ulearn, Xte, Cte, Z, cBase);
        dimW_learned(tr, ii)  = size(Ulearn, 2);

        fprintf('trial %2d/%2d | dim(W)=%.0f | risk_full=%.4g | risk_red=%.4g\n', ...
            tr, cfg.nTrial, dimW_learned(tr,ii), testRisk_full(tr,ii), testRisk_red(tr,ii));
    end
end

% -------------------------- Aggregate mean + 90% CI --------------------
xAxis = cfg.trainSizes;
[mF, ciF] = mean_ci90(log10(testRisk_full + 1e-12));
[mO, ciO] = mean_ci90(log10(testRisk_red  + 1e-12));
[mD, ciD] = mean_ci90(dimW_learned);

% ------------------------------- Plots ---------------------------------
figRisk = figure('Name','Test SPO risk (log10)');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth',1.2);
errorbar(xAxis, mO, ciO, 'LineWidth',1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Supervised SPO+ (full d)', 'Ours: SDS/W + reduced SPO+'}, 'Location','best');
title(sprintf('Shortest path %dx%d, d=%d, target d_*=%d', cfg.g, cfg.g, d, dstar_true));

figDim = figure('Name','dim(W) discovered by SDS');
hold on; grid on; box on;
errorbar(xAxis, mD, ciD, 'LineWidth',1.2);
yline(dstar_true, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('dim(W)');
legend({'Learned dim(W)', 'True d_*'}, 'Location','best');
title('SDS/W dimension detection (compression effect)');

resultsDir = prepare_results_dir();
save_figure(figRisk, fullfile(resultsDir, 'full_dim_corridor_spo_risk.png'));
save_figure(figDim, fullfile(resultsDir, 'full_dim_corridor_dimW.png'));
save(fullfile(resultsDir, 'full_dim_corridor_summary.mat'), ...
    'cfg', 'testRisk_full', 'testRisk_red', 'dimW_learned', ...
    'corrEdges', 'corrPathIdx', 'dstar_true', 'affdimC', 'cBase', 'signalAmp');
fprintf('Saved figures and summary statistics to %s\n', resultsDir);

end

%==========================================================================
%                              FUNCTIONS
%==========================================================================

function [Z, pathSeqs, edge] = build_grid_paths(g)
% Build edge index maps and enumerate all monotone (R,D) paths from (1,1)
% to (g,g).
% Returns:
%   Z: d x K incidence matrix (columns are paths)
%   pathSeqs: 1 x K cell of strings 'R'/'D'
%   edge: struct with fields h, v, d
%
% Edge indexing:
%   Horizontal edges h(i,j): row i=1..g, col j=1..g-1
%   Vertical   edges v(i,j): row i=1..g-1, col j=1..g
%
% Total d = 2*g*(g-1).

    % Horizontal edges
    h = zeros(g, g-1);
    idx = 1;
    for i = 1:g
        for j = 1:(g-1)
            h(i,j) = idx; idx = idx + 1;
        end
    end

    % Vertical edges
    v = zeros(g-1, g);
    for i = 1:(g-1)
        for j = 1:g
            v(i,j) = idx; idx = idx + 1;
        end
    end

    d = idx - 1;
    L = 2*(g-1);
    K = nchoosek(L, g-1);

    Z = zeros(d, K);
    pathSeqs = cell(1, K);

    % Enumerate all sequences with (g-1) rights among L moves
    combs = nchoosek(1:L, g-1);
    for k = 1:size(combs,1)
        seq = repmat('D', 1, L);
        seq(combs(k,:)) = 'R';

        edges = edges_from_seq(seq, h, v, g);
        z = zeros(d,1);
        z(edges) = 1;

        Z(:,k) = z;
        pathSeqs{k} = seq;
    end

    edge.h = h;
    edge.v = v;
    edge.d = d;
end

function edges = edges_from_seq(seq, h, v, g)
% Convert a 'R'/'D' sequence into edge indices.
    r = 1; c = 1;
    L = length(seq);
    edges = zeros(L,1);

    for t = 1:L
        if seq(t) == 'R'
            edges(t) = h(r,c);
            c = c + 1;
        else
            edges(t) = v(r,c);
            r = r + 1;
        end
    end

    if ~(r == g && c == g)
        error('edges_from_seq: invalid path sequence (did not end at (g,g)).');
    end
end

function [corrEdges, corrPathIdx, dstar] = find_corridor_custom_dstar(g, Z, edge, dstar_target)
% Brute-force over subsets of the (g-1)^2 unit squares.
% Corridor edges := base path edges + boundary edges of chosen squares.
% Then corridor-path set := all paths whose edges are subset of corridor edges.
% We pick the narrowest corridor among those achieving dstar_target:
%   minimize (#edges), then (#corridor paths).

    if g ~= 4
        error('This helper is written/tested for g=4. Extend if needed.');
    end

    h = edge.h;
    v = edge.v;
    d = edge.d;
    K = size(Z,2);

    % Base path (always included): RRRDDD
    baseSeq   = [repmat('R',1,g-1), repmat('D',1,g-1)];
    baseEdges = edges_from_seq(baseSeq, h, v, g);

    nSq = (g-1)*(g-1);
    bestFound = false;
    bestNumEdges = inf;
    bestNumPaths = inf;

    corrEdges = [];
    corrPathIdx = [];
    dstar = -1;

    for mask = 0:(2^nSq - 1)
        E = baseEdges;

        % decode square mask (bit order: i major, then j)
        bit = 1;
        for i = 1:(g-1)
            for j = 1:(g-1)
                if bitget(mask, bit)
                    % square (i,j) has top-left corner at (i,j)
                    E = [E; h(i,j); h(i+1,j); v(i,j); v(i,j+1)]; %#ok<AGROW>
                end
                bit = bit + 1;
            end
        end

        E = unique(E);

        % Which paths lie entirely in E?
        EMask = false(d,1);
        EMask(E) = true;

        inCorr = false(1, K);
        for k = 1:K
            used = find(Z(:,k) > 0.5);
            if all(EMask(used))
                inCorr(k) = true;
            end
        end
        P = find(inCorr);

        if isempty(P)
            continue;
        end

        % d_* = rank of differences among corridor paths
        if numel(P) == 1
            r = 0;
        else
            W = Z(:,P) - Z(:,P(1));
            r = rank(W, 1e-12);
        end

        if r ~= dstar_target
            continue;
        end

        % narrowness criterion
        if (numel(E) < bestNumEdges) || (numel(E) == bestNumEdges && numel(P) < bestNumPaths)
            bestFound = true;
            bestNumEdges = numel(E);
            bestNumPaths = numel(P);
            corrEdges = E;
            corrPathIdx = P;
            dstar = r;
        end
    end

    if ~bestFound
        error('No corridor found achieving d_*=%d. (Unexpected for 4x4; should be possible.)', dstar_target);
    end
end

function verify_domination(Z, corrPathIdx, lbC, ubC)
% Check: max corridor-path cost (under ubC) < min outside-path cost (under lbC)
% This implies: for any c in C, the optimal path must lie in the corridor.

    K = size(Z,2);
    inCorr = false(1,K);
    inCorr(corrPathIdx) = true;

    % Upper bound on corridor costs
    maxCorr = -inf;
    for k = corrPathIdx(:)'
        maxCorr = max(maxCorr, ubC' * Z(:,k));
    end

    % Lower bound on non-corridor costs
    minOut = inf;
    outIdx = find(~inCorr);
    if ~isempty(outIdx)
        for k = outIdx(:)'
            minOut = min(minOut, lbC' * Z(:,k));
        end
    end

    fprintf('Domination check: maxCorr(ub) = %.2f, minOutside(lb) = %.2f.\n', maxCorr, minOut);
    if maxCorr >= minOut
        warning('Domination check FAILED: outside paths may become optimal. Consider increasing the outside base.');
    else
        fprintf('Domination check OK: outside paths are never optimal for any c in C.\n');
    end
end

function [X, C] = sample_contextual_costs(n, p, cBase, Ustar, Atrue, signalAmp, noiseCorr, noiseOut, lbC, ubC, corrEdges)
% Generate synthetic (x,c) samples:
%   x ~ N(0,I_p)
%   c = cBase + Ustar * g(x) + noise, then clipped into [lbC,ubC].
%
% The contextual signal only lives in span(Ustar), so the intrinsic
% dimension is dim(Ustar)=d_*.

    d = length(cBase);

    X = randn(n, p);

    % latent coordinates g(x) in [-1,1]^r
    G = tanh((Atrue * X') / sqrt(p)); % r x n

    % contextual signal in R^d
    Sig = signalAmp * (Ustar * G);    % d x n

    % noise (full-dimensional, but outside is decision-irrelevant by domination)
    Noise = noiseOut * (2*rand(d,n) - 1);
    Noise(corrEdges,:) = noiseCorr * (2*rand(numel(corrEdges), n) - 1);

    Cmat = cBase + Sig + Noise;
    Cmat = min(max(Cmat, lbC), ubC);  % clip into box

    C = Cmat'; % n x d
end

function idx = oracle_path_idx(c, Z)
% Return argmin_k c^T z_k (z_k = column k of Z).
    costs = Z' * c;
    [~, idx] = min(costs);
end

function risk = mean_spo_risk_full(B, X, C, Z, cBase)
    n = size(X,1);
    riskVec = zeros(n,1);

    for i = 1:n
        phi = [X(i,:)'; 1];
        c = C(i,:)';
        chat = cBase + B * phi;

        kTrue = oracle_path_idx(c, Z);
        kHat  = oracle_path_idx(chat, Z);

        riskVec(i) = c' * (Z(:,kHat) - Z(:,kTrue));
    end
    risk = mean(riskVec);
end

function risk = mean_spo_risk_reduced(G, U, X, C, Z, cBase)
    n = size(X,1);
    riskVec = zeros(n,1);

    for i = 1:n
        phi = [X(i,:)'; 1];
        c = C(i,:)';
        chat = cBase + U * (G * phi);

        kTrue = oracle_path_idx(c, Z);
        kHat  = oracle_path_idx(chat, Z);

        riskVec(i) = c' * (Z(:,kHat) - Z(:,kTrue));
    end
    risk = mean(riskVec);
end

function B = train_spoplus_full(X, C, Z, cBase, numEpochs, lr)
% Adam on SPO+ subgradients for linear predictor chat = cBase + B*[x;1].

    [n, p] = size(X);
    d = size(Z,1);
    B = zeros(d, p+1);

    m = zeros(size(B));
    v = zeros(size(B));
    beta1 = 0.9;
    beta2 = 0.999;
    epsVal = 1e-8;
    t = 0;

    for ep = 1:numEpochs %#ok<NASGU>
        perm = randperm(n);
        for ii = 1:n
            t = t + 1;
            i = perm(ii);

            phi = [X(i,:)'; 1];
            c   = C(i,:)';
            chat = cBase + B * phi;

            k0 = oracle_path_idx(c, Z);          % x*(c)
            k1 = oracle_path_idx(2*chat - c, Z); % x*(2 chat - c)

            gc = 2 * (Z(:,k0) - Z(:,k1));        % subgradient wrt chat
            gB = gc * (phi');                    % chain rule

            % Adam update
            m = beta1*m + (1-beta1)*gB;
            v = beta2*v + (1-beta2)*(gB.^2);
            mhat = m / (1 - beta1^t);
            vhat = v / (1 - beta2^t);

            B = B - lr * mhat ./ (sqrt(vhat) + epsVal);
        end
    end
end

function G = train_spoplus_reduced(X, C, Z, cBase, U, numEpochs, lr)
% Adam on SPO+ subgradients for compressed predictor chat = cBase + U*(G*[x;1]).

    [n, p] = size(X);
    r = size(U,2);
    G = zeros(r, p+1);

    m = zeros(size(G));
    v = zeros(size(G));
    beta1 = 0.9;
    beta2 = 0.999;
    epsVal = 1e-8;
    t = 0;

    for ep = 1:numEpochs %#ok<NASGU>
        perm = randperm(n);
        for ii = 1:n
            t = t + 1;
            i = perm(ii);

            phi = [X(i,:)'; 1];
            c   = C(i,:)';
            chat = cBase + U * (G * phi);

            k0 = oracle_path_idx(c, Z);
            k1 = oracle_path_idx(2*chat - c, Z);

            gc = 2 * (Z(:,k0) - Z(:,k1));   % subgradient wrt chat
            gG = (U' * gc) * (phi');        % chain rule

            % Adam update
            m = beta1*m + (1-beta1)*gG;
            v = beta2*v + (1-beta2)*(gG.^2);
            mhat = m / (1 - beta1^t);
            vhat = v / (1 - beta2^t);

            G = G - lr * mhat ./ (sqrt(vhat) + epsVal);
        end
    end
end

function U = learn_sds_basis(Ctr, Z, lbC, ubC, tol)
% Repository SDS learner for the discrete path set.
% For each observed cost c, check if the current directions U certify
% pointwise sufficiency; if not, add one violated direction q = z_alt - z_star.

    d = size(Z,1);
    K = size(Z,2);

    % linprog options
    opts = [];
    if exist('optimoptions','file') == 2
        opts = optimoptions('linprog','Display','none');
    elseif exist('optimset','file') == 2
        opts = optimset('Display','none'); %#ok<OPTIMSET>
    end

    U = zeros(d,0);
    D = zeros(d,0); % store raw query directions

    n = size(Ctr,1);
    for i = 1:n
        c = Ctr(i,:)';

        kStar = oracle_path_idx(c, Z);
        zStar = Z(:,kStar);

        % Find most violating alternative under the fiber constraints U^T c' = U^T c
        bestVal = +inf;
        bestQ = [];

        for k = 1:K
            if k == kStar, continue; end
            q = Z(:,k) - zStar;

            val = min_q_over_fiber(q, U, c, lbC, ubC, opts);
            if val < bestVal
                bestVal = val;
                bestQ = q;
            end
        end

        if bestVal < -tol && ~isempty(bestQ)
            % Add only if linearly independent of current U
            if isempty(U)
                qPerp = bestQ;
            else
                qPerp = bestQ - U*(U'*bestQ);
            end
            if norm(qPerp) > 1e-10
                D = [D, bestQ]; %#ok<AGROW>
                U = orth(D);
            end
        end
    end
end

function val = min_q_over_fiber(q, U, c, lbC, ubC, opts)
% Solve:  min_{c' in [lbC,ubC], U^T c' = U^T c}  q^T c'
% If U is empty, closed form over a box.

    if isempty(U)
        % Min over box: choose lb where q>0 and ub where q<0
        val = sum(max(q,0).*lbC + min(q,0).*ubC);
        return;
    end

    f = q;
    A = [];
    b = [];
    Aeq = U';
    beq = U'*c;

    try
        if isempty(opts)
            [~, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lbC, ubC);
        else
            [~, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lbC, ubC, opts);
        end
    catch ME
        error('learn_sds_basis:linprogFailed', ...
            'linprog failed while learning the SDS basis: %s', ME.message);
    end

    if exitflag <= 0
        msg = '';
        if isstruct(output) && isfield(output, 'message')
            msg = output.message;
        end
        error('learn_sds_basis:linprogExitflag', ...
            'linprog returned exitflag %d while learning the SDS basis. %s', exitflag, msg);
    end

    val = fval;
end

%==========================================================================
%                     Mean and 90% CI helper
%==========================================================================

function [m, ci] = mean_ci90(M)
% Mean and 90% CI half-width across trials (rows), ignoring NaNs.
z = 1.645;
m = mean(M, 1, 'omitnan');
nEff = sum(~isnan(M), 1);
sd = std(M, 0, 1, 'omitnan');
se = sd ./ max(sqrt(nEff), 1);
ci = z * se;
end

function resultsDir = prepare_results_dir()
repoDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoDir, 'results');
if exist(resultsDir, 'dir') ~= 7
    mkdir(resultsDir);
end
end

function save_figure(figHandle, filePath)
if exist('exportgraphics', 'file') == 2
    exportgraphics(figHandle, filePath, 'Resolution', 300);
else
    saveas(figHandle, filePath);
end
end

