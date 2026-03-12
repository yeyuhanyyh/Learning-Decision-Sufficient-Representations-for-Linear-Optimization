# Learning Decision-Sufficient Representations for Linear Optimization

This repository provides compact MATLAB implementations of two shortest-path experiments illustrating model compression for contextual linear optimization (CLO).

The experiments compare:

1. full-dimensional SPO+ training in the ambient cost space;
2. reduced SPO+ training after learning a decision-relevant subspace.

The central message is that when optimal decisions depend only on a low-dimensional decision-relevant subspace, predictor training can scale with `d_*` rather than the ambient dimension `d`.

## Project Overview

This repository contains two experiments:

| Experiment | File | Purpose |
| --- | --- | --- |
| Low-affine-dimension box experiment | `codes/Version_1_C_with_low_aff_dimension.m` | Warm-up setting where only a small subset of cost coordinates can vary. |
| Full-dimensional corridor experiment | `codes/Version_2_structured_full_dimension_C.m` | Main compression setting where the prior cost set is still full-dimensional, but only a narrow family of paths can ever be optimal. |

Both experiments follow the same two-stage idea:

- **Stage I:** learn a low-dimensional subspace `W_hat` from observed costs;
- **Stage II:** train a contextual SPO+ predictor inside `W_hat` instead of in the full ambient space.

## Key CLO Formulas

Let `x*(v) ∈ argmin_{x ∈ X} v^T x` be a fixed deterministic oracle.

- Ellipsoidal prior:
  `C = { c ∈ R^d : (c - c0)^T Σ^{-1} (c - c0) ≤ 1 }`

- Canonical lifting:
  `L_U = Σ U (U^T Σ U)^(-1)`

  `lift_U(s) = c0 + L_U s`

- SPO loss:
  `ℓ_SPO(c_hat, c) = c^T x*(c_hat) - c^T x*(c)`

- SPO risk:
  `R_SPO(f) = E[ ℓ_SPO(f(ξ), c) ]`

- SPO+ surrogate:
  `ℓ_SPO+(c_hat, c) = max_{x ∈ X} (c - 2 c_hat)^T x + 2 c_hat^T x*(c) - c^T x*(c)`

- A valid SPO+ subgradient:
  `2 ( x*(c) - x*(2 c_hat - c) )`

- Compressed predictor:
  `c_hat_theta(ξ) = lift_{U*}(g_theta(ξ)) = c0 + L_{U*} g_theta(ξ)`

- Linear coordinate model:
  `g_theta(ξ) = B_theta ξ`, with `B_theta ∈ R^(d_* × p)`

  This reduces the number of trainable parameters from `d p` to `d_* p`.

- Centered Stage-I conditional-mean model:
  `c - c0 = A_mu ξ + ε`, `E[ε | ξ] = 0`, `μ(ξ) = c0 + A_mu ξ`

- Lifted Stage-I predictor:
  `mu_tilde(ξ) = lift_{U_hat}(U_hat^T(mu_hat(ξ) - c0))`
  
  equivalently,
  
  `mu_tilde(ξ) = c0 + L_{U_hat} U_hat^T (mu_hat(ξ) - c0)`

- Stage-II generalization bound in the compressed class:
  `R_SPO(f) ≤ R_hat_SPO(f) + 2 ω_X(C) sqrt( 2(d_* p + 1) log(n |X^∠|^2) / n ) + ω_X(C) sqrt( log(1/δ) / (2n) )`

So the dominant dimension term depends on `d_*`, not `d`.

## Repository Structure

```text
.
├── codes
│   ├── Version_1_C_with_low_aff_dimension.m
│   ├── Version_2_structured_full_dimension_C.m
│   └── results/   % created automatically after running the scripts
├── Learning_Decision_Sufficient_Representations_for_Linear_Optimization_arxiv.pdf
└── README.md
```

## Requirements

- MATLAB R2021a or later
- Optimization Toolbox (`linprog`)

No external dataset is required.

## How to Run

Open MATLAB in the `codes/` folder, then open and run either of the following files from the editor:

- `Version_1_C_with_low_aff_dimension.m`
- `Version_2_structured_full_dimension_C.m`

Each script is self-contained and will automatically create a `results/` folder inside `codes/`.

## Output Files

After running the experiments, the following files are generated under `codes/results/`:

| Experiment | Figure 1 | Figure 2 | Summary |
| --- | --- | --- | --- |
| Low-affine-dimension box | `low_affdim_spo_risk.png` | `low_affdim_dimW.png` | `low_affdim_summary.mat` |
| Full-dimensional corridor | `full_dim_corridor_spo_risk.png` | `full_dim_corridor_dimW.png` | `full_dim_corridor_summary.mat` |

## Sample Result

Each experiment produces two main outputs:

1. **Test SPO risk**
   - compares full-dimensional SPO+ against reduced SPO+;
   - reported on a log scale.

2. **Learned dimension of the subspace**
   - tracks the discovered `dim(W)` as the labeled sample size increases.

Typical qualitative behavior:

- the learned dimension stabilizes quickly;
- the reduced model becomes competitive with fewer labeled samples;
- the full-dimensional corridor experiment is the clearest model-compression example, because `affdim(C) = d` while the decision-relevant dimension remains much smaller.

## Notes

- `Version_1` is the simpler warm-up experiment.
- `Version_2` is the main full-dimensional compression experiment.
- The code is intended as a compact numerical illustration of decision-sufficient representation learning and contextual linear optimization, not as a full reproduction package for every theorem in the paper.

## Author

Yuhan Ye
