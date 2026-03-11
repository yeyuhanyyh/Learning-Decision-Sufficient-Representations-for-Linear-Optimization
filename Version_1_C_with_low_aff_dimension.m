function initial_C_low_aff_dimension()
%==========================================================================
% Compact public-release implementation of the low-affine-dimension shortest
% path experiment used as a reader-facing reference.
%
% We compare:
%   (1) Supervised SPO+ with a full-dimensional linear predictor
%   (2) Online SDS/W learning plus a reduced SPO+ predictor
%
% Key design goal:
%   Construct an instance with d_star << d by restricting the cost set C so
%   that only r coordinates can vary and all other coordinates stay fixed.
%
% Outputs:
%   - Figure 1: log10(Test SPO risk) vs #labeled samples (mean +/- 90% CI)
%   - Figure 2: mean dim(W) discovered by SDS vs #labeled samples
%   - Saved PNG figures and a MAT summary under results/
%
% Requires Optimization Toolbox for linprog (used in the SDS check).
%==========================================================================

clc; close all;

if exist('linprog','file') ~= 2
    error('This script requires Optimization Toolbox (linprog).');
end

%----------------------------- Configuration ------------------------------
cfg.seed       = 20260311;  % global seed used for setup and per-trial seeds
cfg.gridSize   = 5;         % 5x5 node grid (4 moves in each direction)
cfg.p          = 5;         % feature dimension
cfg.Ntrain     = 300;       % number of labeled training samples (all labeled)
cfg.Ntest      = 2000;      % test set size
cfg.nTrials    = 10;        % repeat trials (increase to 25 if you want)
cfg.evalEvery  = 1;         % evaluate every sample (can set to 5/10 for speed)

% Cost-set design: only r coords vary => intrinsic dimension <= r
cfg.r_true     = 5;         % desired small intrinsic dimension d_star
cfg.c0         = 10.0;      % baseline constant cost on all (non-varying) edges
cfg.deltaScale = 1;         % variation amplitude on varying edges
cfg.noiseScale = 0.05;      % small noise to break ties
cfg.marginTol  = 1e-3;      % strictness in SDS: require alt path better by >= marginTol

% Training (mini-batch SGD on SPO+)
cfg.batchSize  = 25;        % mini-batch size
cfg.updatesPerSample = 4;   % #SGD updates after each new sample arrives
cfg.lr_full0   = 0.20;      % base LR full
cfg.lr_red0    = 0.25;      % base LR reduced
cfg.gradClip   = 5.0;       % gradient clipping (Fro norm)

lpOpts = optimoptions('linprog','Display','none');
rng(cfg.seed, 'twister');

fprintf('=== SDS/W compression vs Supervised-SPO+ (Shortest Path %dx%d) ===\n', cfg.gridSize, cfg.gridSize);

%-------------------- Build decision set: all monotone paths --------------
[Zbase, edgeInfo] = buildMonotonePathIncidence(cfg.gridSize); %#ok<ASGLU>
[dEdge, K] = size(Zbase);
fprintf('Monotone paths K=%d, edge-dimension d=%d\n', K, dEdge);

% Choose r varying edges so that rank(Z(varEdges,:)) = r (greedy)
varEdges = choose_var_edges_with_rank(Zbase, cfg.r_true);
rEff = rank(Zbase(varEdges,:), 1e-12);

fprintf('Chosen varying edges |I|=%d, rank(Z(I,:))=%d\n', numel(varEdges), rEff);
fprintf('Target intrinsic dim d_star ~ %d  (and d=%d)\n', numel(varEdges), dEdge);

% Prior cost set C as a box with fixed coordinates
lbC = cfg.c0 * ones(dEdge,1);
ubC = cfg.c0 * ones(dEdge,1);
rad = cfg.deltaScale + cfg.noiseScale + 0.10;  % widen slightly for feasibility
lbC(varEdges) = cfg.c0 - rad;
ubC(varEdges) = cfg.c0 + rad;

% varIdx used to ensure SDS directions live only on varying coords
varIdx = find(ubC > lbC + 1e-12);
r_true = numel(varIdx);

% Storage: risks and dim(W)
Tmax = cfg.Ntrain;
risk_full_all = nan(cfg.nTrials, Tmax);
risk_ours_all = nan(cfg.nTrials, Tmax);
dimW_all      = nan(cfg.nTrials, Tmax);

%------------------------------- Trials -----------------------------------
for tr = 1:cfg.nTrials
    rng(cfg.seed + tr, 'twister');
    fprintf('\n--- Trial %d/%d ---\n', tr, cfg.nTrials);

    % Random coefficients for varying edges (p x r_true)
    A = randn(cfg.p, r_true);

    % Sample train/test data from the restricted C (only varEdges vary)
    [Xtrain, Ctrain] = sample_low_intrinsic_costs(cfg.Ntrain, cfg.p, dEdge, varEdges, cfg, A);
    [Xtest,  Ctest ] = sample_low_intrinsic_costs(cfg.Ntest,  cfg.p, dEdge, varEdges, cfg, A);

    % Initialize full predictor: chat = B*[x;1], B is d x (p+1)
    Bfull = zeros(dEdge, cfg.p+1);

    % Initialize our compressed predictor: chat = U*(G*[x;1])
    U = zeros(dEdge,0);            % orthonormal basis for W (columns)
    G = zeros(0, cfg.p+1);         % reduced parameter matrix (t x (p+1))

    for t = 1:Tmax
        % Add one labeled sample (we assume full supervision stream)
        c_t = Ctrain(t,:)';  % column

        % ----- Our SDS/W update (online) -----
        U = sds_pointwise_update_box(U, c_t, Zbase, lbC, ubC, varIdx, cfg.marginTol, lpOpts);

        % expand G if dim(W) increased
        tDim = size(U,2);
        if size(G,1) < tDim
            G = [G; zeros(tDim - size(G,1), cfg.p+1)]; %#ok<AGROW>
        end

        % ----- Training updates (mini-batch SGD) -----
        lrFull = cfg.lr_full0 / sqrt(max(t,1));
        lrRed  = cfg.lr_red0  / sqrt(max(t,1));

        for u = 1:cfg.updatesPerSample %#ok<NASGU>
            Bfull = sgd_full_spoplus_step(Bfull, Xtrain, Ctrain, t, cfg.batchSize, Zbase, lrFull, cfg.gradClip);
            if tDim > 0
                G = sgd_reduced_spoplus_step(G, U, Xtrain, Ctrain, t, cfg.batchSize, Zbase, lrRed, cfg.gradClip);
            end
        end

        % ----- Evaluate -----
        if mod(t, cfg.evalEvery) == 0
            risk_full_all(tr,t) = test_spo_risk_full(Bfull, Xtest, Ctest, Zbase);
            risk_ours_all(tr,t) = test_spo_risk_reduced(G, U, Xtest, Ctest, Zbase);
            dimW_all(tr,t)      = size(U,2);
        end
    end

    finalDim = dimW_all(tr, find(~isnan(dimW_all(tr,:)), 1, 'last'));
    if isempty(finalDim), finalDim = size(U,2); end
    fprintf('Trial %d: final dim(W)=%d (true varying coords=%d)\n', tr, finalDim, r_true);
end

%------------------------ Aggregate mean + 90% CI --------------------------
xAxis = 1:Tmax;
[mF, ciF] = mean_ci90(log10(risk_full_all + 1e-12));
[mO, ciO] = mean_ci90(log10(risk_ours_all + 1e-12));

% For dim(W): we do NOT plot CI; only mean over trials.
mD = mean(dimW_all, 1, 'omitnan');           % 1 x Tmax

% Print parameter compression
paramFull = dEdge*(cfg.p+1);
lastEvalIdx = find(~all(isnan(dimW_all),1), 1, 'last');
if isempty(lastEvalIdx)
    meanFinalDim = mean(dimW_all(:), 'omitnan');
else
    meanFinalDim = mean(dimW_all(:, lastEvalIdx), 'omitnan');
end
paramRed_final = round(meanFinalDim * (cfg.p+1));
fprintf('\n=== Compression summary ===\n');
fprintf('Ambient d=%d, p=%d => full linear params = d*(p+1) = %d\n', dEdge, cfg.p, paramFull);
fprintf('Mean final dim(W) across trials = %.2f\n', meanFinalDim);
fprintf('Reduced params using final dim(W) ~ %d\n', paramRed_final);

%------------------------------- Plots ------------------------------------
figRisk = figure('Name','Test SPO risk (log10)');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth',1.2);
errorbar(xAxis, mO, ciO, 'LineWidth',1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Supervised SPO+ with full d', 'Ours: dimension reduced SPO+'}, 'Location','best');
title(sprintf('Shortest path %dx%d, d=%d, target d_*=%d', cfg.gridSize, cfg.gridSize, dEdge, r_true));

% ---- Figure 2: mean dim(W), start at (0,0), only show first ~10 samples ----
xAxisD = 0:Tmax;                     % include x=0
mD0    = [0, mD];                    % prepend (0,0) point

xMaxShow = min(10, Tmax);
idxShow  = (xAxisD <= xMaxShow);

figDim = figure('Name','dim(W) discovered by SDS (mean only, early)');
hold on; grid on; box on;
plot(xAxisD(idxShow), mD0(idxShow), '-o', 'LineWidth', 1.6, 'MarkerSize', 4);
yline(r_true, '--', 'LineWidth', 1.2);

xlim([0, xMaxShow]);
ylim([0, r_true + 0.5]);             % show growth clearly from 0 upward

xlabel('# labeled training samples');
ylabel('dim(W) (mean over trials)');
legend({'Mean learned dim(W)', 'Upper bound on d_*'}, 'Location','best');
title('SDS/W dimension detection (early-stage)');

resultsDir = prepare_results_dir();
save_figure(figRisk, fullfile(resultsDir, 'low_affdim_spo_risk.png'));
save_figure(figDim, fullfile(resultsDir, 'low_affdim_dimW.png'));
save(fullfile(resultsDir, 'low_affdim_summary.mat'), ...
    'cfg', 'varEdges', 'risk_full_all', 'risk_ours_all', 'dimW_all', ...
    'mF', 'ciF', 'mO', 'ciO', 'mD', 'r_true', 'paramFull', 'paramRed_final');
fprintf('Saved figures and summary statistics to %s\n', resultsDir);

end

%==========================================================================
%                         Data generation (low intrinsic)
%==========================================================================

function [X, C] = sample_low_intrinsic_costs(n, p, d, varEdges, cfg, A)
% Generate contexts X and costs C in a low-dimensional box:
%   - For j not in varEdges: c_j = c0 (fixed)
%   - For j in varEdges:    c_j = c0 + deltaScale*tanh(a_j^T x / sqrt(p)) + noise

X = randn(n,p);
C = cfg.c0 * ones(n,d);

r = numel(varEdges);
for i = 1:n
    x = X(i,:)';
    for j = 1:r
        t = (A(:,j)' * x) / sqrt(p);
        delta = cfg.deltaScale * tanh(t);
        noise = cfg.noiseScale * (2*rand()-1);
        C(i, varEdges(j)) = cfg.c0 + delta + noise;
    end
end

end

%==========================================================================
%                     Training: SPO+ mini-batch SGD
%==========================================================================

function B = sgd_full_spoplus_step(B, X, C, m, batchSize, Z, lr, gradClip)
% One mini-batch SGD step for full model: chat = B*[x;1]
[d, p1] = size(B); %#ok<ASGLU>

bs = min(batchSize, m);
idx = randi(m, [bs,1]);

Grad = zeros(size(B));
for k = 1:bs
    i = idx(k);
    xbar = [X(i,:)'; 1];       % (p+1)x1
    chat = B * xbar;           % d x 1
    ctrue = C(i,:)';           % d x 1
    subg = spoplus_subgrad(chat, ctrue, Z); % d x 1
    Grad = Grad + subg * xbar';
end
Grad = Grad / bs;

gn = norm(Grad(:),2);
if gn > gradClip
    Grad = Grad * (gradClip / gn);
end
B = B - lr * Grad;
end

function G = sgd_reduced_spoplus_step(G, U, X, C, m, batchSize, Z, lr, gradClip)
% One mini-batch SGD step for reduced model: chat = U*(G*[x;1])

bs = min(batchSize, m);
idx = randi(m, [bs,1]);

Grad = zeros(size(G));
for k = 1:bs
    i = idx(k);
    xbar = [X(i,:)'; 1];
    ghat = G * xbar;      % t x 1
    chat = U * ghat;      % d x 1
    ctrue = C(i,:)';
    subg = spoplus_subgrad(chat, ctrue, Z);  % d x 1
    Grad = Grad + (U' * subg) * xbar';
end
Grad = Grad / bs;

gn = norm(Grad(:),2);
if gn > gradClip
    Grad = Grad * (gradClip / gn);
end
G = G - lr * Grad;
end

%==========================================================================
%                       SPO / SPO+ primitives
%==========================================================================

function loss = spo_loss(chat, ctrue, Z)
chat = chat(:);
ctrue = ctrue(:);
[~, w_hat] = oracle_path(chat, Z);
[~, w_opt] = oracle_path(ctrue, Z);
loss = ctrue' * (w_hat - w_opt);
end

function subg = spoplus_subgrad(chat, ctrue, Z)
% SPO+ subgradient wrt chat: 2*(w*(c) - w*(2chat - c))
chat  = chat(:);
ctrue = ctrue(:);
[~, w_opt] = oracle_path(ctrue, Z);
[~, w_adv] = oracle_path(2*chat - ctrue, Z);
subg = 2 * (w_opt - w_adv);
end

function [idx, w] = oracle_path(c, Z)
% Oracle for finite path set: minimize c^T w over columns of Z
c = c(:);                 % critical: avoid row/col bugs
vals = Z' * c;            % K x 1
[~, idx] = min(vals);
w = Z(:, idx);
end

%==========================================================================
%                     Evaluation: test SPO risk
%==========================================================================

function r = test_spo_risk_full(B, Xtest, Ctest, Z)
n = size(Xtest,1);
tot = 0;
for i = 1:n
    xbar = [Xtest(i,:)'; 1];
    chat = B * xbar;
    ctrue = Ctest(i,:)';
    tot = tot + spo_loss(chat, ctrue, Z);
end
r = tot / n;
end

function r = test_spo_risk_reduced(G, U, Xtest, Ctest, Z)
n = size(Xtest,1);
tot = 0;
for i = 1:n
    xbar = [Xtest(i,:)'; 1];
    chat = U * (G * xbar);
    ctrue = Ctest(i,:)';
    tot = tot + spo_loss(chat, ctrue, Z);
end
r = tot / n;
end

%==========================================================================
%              Our SDS: pointwise sufficiency update (box prior)
%==========================================================================

function U = sds_pointwise_update_box(U, c, Z, lbC, ubC, varIdx, marginTol, lpOpts)
% Simplified pointwise routine for the finite-path setting:
% Check if there exists c' in [lbC,ubC] with same measurements U'*c'=U'*c
% such that some other path becomes strictly better than the current optimal path.
% If yes, add a new direction (restricted to varIdx) and repeat.

c = c(:);
[d, K] = size(Z); %#ok<ASGLU>
[idxStar, wStar] = oracle_path(c, Z);

while true
    m = U' * c;  % current measurements

    found = false;
    altIdx = -1;

    for k = 1:K
        if k == idxStar, continue; end

        % Feasibility LP:
        %   find cvar s.t.
        %     (Z(:,k)-wStar)' cvar <= -marginTol
        %     U' cvar = m
        %     lbC <= cvar <= ubC
        f = zeros(d,1);
        Aineq = (Z(:,k) - wStar)';  % 1 x d
        bineq = -marginTol;

        if isempty(U)
            Aeq = [];
            beq = [];
        else
            Aeq = U';
            beq = m;
        end

        [~, ~, exitflag] = linprog(f, Aineq, bineq, Aeq, beq, lbC, ubC, lpOpts);
        if exitflag > 0
            found = true;
            altIdx = k;
            break;
        end
    end

    if ~found
        break; % sufficient for this c under current U
    end

    % New direction: difference between alt path and current optimal path,
    % restricted to the variable coordinates (since fixed coords cannot vary).
    qNew = zeros(d,1);
    qNew(varIdx) = Z(varIdx, altIdx) - wStar(varIdx);

    % Orthonormalize and append
    if isempty(U)
        qOrth = qNew;
    else
        qOrth = qNew - U * (U' * qNew);
    end

    nrm = norm(qOrth,2);
    if nrm < 1e-10
        break; % numerically no new direction
    end
    U = [U, qOrth / nrm]; %#ok<AGROW>
end

end

%==========================================================================
%                    Build monotone paths (grid)
%==========================================================================

function [Z, edgeInfo] = buildMonotonePathIncidence(g)
% Monotone paths from (1,1) to (g,g) using only East/North.
% Edge indexing:
%   horizontals: r=1..g, c=1..g-1     => idxH=(r-1)*(g-1)+c
%   verticals:   r=1..g-1, c=1..g     => idxV=g*(g-1)+(c-1)*(g-1)+r
L = 2*(g-1);
K = nchoosek(L, g-1);
d = 2*g*(g-1);

Z = zeros(d, K);
comb = nchoosek(1:L, g-1); % positions of E moves

for k = 1:K
    Epos = comb(k,:);
    move = repmat('N', 1, L);
    move(Epos) = 'E';

    r = 1; c = 1;
    z = zeros(d,1);

    for t = 1:L
        if move(t) == 'E'
            idxH = (r-1)*(g-1) + c;
            z(idxH) = z(idxH) + 1;
            c = c + 1;
        else
            idxV = g*(g-1) + (c-1)*(g-1) + r;
            z(idxV) = z(idxV) + 1;
            r = r + 1;
        end
    end

    Z(:,k) = z;
end

edgeInfo = struct('g',g,'d',d,'K',K);
end

%==========================================================================
%           Choose varying edges with full rank on Z(varEdges,:)
%==========================================================================

function varEdges = choose_var_edges_with_rank(Z, rTarget)
d = size(Z,1);
varEdges = [];
curRank = 0;

cands = randperm(d);
for idx = cands
    newSet = [varEdges, idx];
    newRank = rank(Z(newSet,:), 1e-12);
    if newRank > curRank
        varEdges = newSet;
        curRank = newRank;
        if curRank >= rTarget
            break;
        end
    end
end

if curRank < rTarget
    error('Cannot reach rank %d with available edges. Reduce cfg.r_true.', rTarget);
end
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



