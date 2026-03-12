# Learning Decision-Sufficient Representations for Linear Optimization

This repository contains MATLAB code for the numerical shortest-path experiments in the paper **Learning Decision-Sufficient Representations for Linear Optimization**.

The code illustrates the following idea: if optimal decisions depend only on a low-dimensional decision-relevant subspace, then a contextual predictor can be trained in that learned reduced space instead of in the full ambient cost space.

---

## Project Overview

This repository includes two synthetic shortest-path experiments.

| Experiment | File | Description |
| --- | --- | --- |
| Low-affine-dimension setting | `codes/Version_1_C_with_low_aff_dimension.m` | A warm-up experiment where only a low-affine-complexity part of the cost vector varies. |
| Full-dimensional structured setting | `codes/Version_2_structured_full_dimension_C.m` | The main compression experiment, where the ambient cost space is full-dimensional but the decision-relevant structure is low-dimensional. |

In both experiments, the workflow is the same:

1. learn a low-dimensional subspace `W_hat` from observed costs;
2. train a reduced SPO+ predictor inside `W_hat`;
3. compare it against a standard full-dimensional SPO+ baseline.

---

## Method

### Contextual Linear Optimization

For a cost vector `c`, the decision rule is

```text
x*(c) ∈ argmin_{x ∈ X} c^T x
```

where `X` is the feasible set of shortest-path decisions.

### Prior Set and Canonical Lifting

The contextual LO formulation in the paper uses the ellipsoidal prior

```text
C = { c ∈ R^d : (c - c0)^T Sigma^{-1} (c - c0) ≤ 1 }
```

For a basis `U` of a decision-relevant subspace, define the canonical lifting map

```text
L_U = Sigma U (U^T Sigma U)^{-1}
lift_U(s) = c0 + L_U s
```

This map converts a low-dimensional coordinate back to a feasible cost vector in the original ambient cost space.

### Stage I: Learn a Decision-Sufficient Representation

From contextual samples `(xi, c)`, estimate the conditional mean `mu(xi) = E[c | xi]`.  
A centered linear model is

```text
c - c0 = A_mu xi + eps,    E[eps | xi] = 0
mu(xi) = c0 + A_mu xi
```

After learning a subspace `W_hat` with orthonormal basis `U_hat`, the lifted compressed predictor is

```text
mu_tilde(xi) = lift_{U_hat}(U_hat^T (mu_hat(xi) - c0))
             = c0 + L_{U_hat} U_hat^T (mu_hat(xi) - c0)
```

### Stage II: Train SPO+ in the Learned Subspace

Given a predicted cost `c_hat`, the SPO loss is

```text
ell_SPO(c_hat, c) = c^T x*(c_hat) - c^T x*(c)
```

Its convex surrogate is

```text
ell_SPO+(c_hat, c) = max_{x ∈ X} (c - 2 c_hat)^T x + 2 c_hat^T x*(c) - c^T x*(c)
```

A valid SPO+ subgradient is

```text
2 ( x*(c) - x*(2 c_hat - c) )
```

The compressed contextual predictor has the form

```text
c_hat_theta(xi) = lift_{U_star}(g_theta(xi))
                = c0 + L_{U_star} g_theta(xi)
```

For a linear coordinate model,

```text
g_theta(xi) = B_theta xi,    B_theta ∈ R^{d_star × p}
```

so the number of trainable parameters is reduced from `d p` to `d_star p`.

---

## Repository Structure

```text
Learning-Decision-Sufficient-Representations-for-Linear-Optimization/
├── codes/
│   ├── Version_1_C_with_low_aff_dimension.m
│   ├── Version_2_structured_full_dimension_C.m
│   └── results/   % generated automatically after running the code
├── Learning_Decision_Sufficient_Representations_for_Linear_Optimization_arxiv.pdf
└── README.md
```

---

## Requirements

- MATLAB
- Optimization Toolbox (`linprog`)

No external dataset is required.

---

## How to Run

1. Open MATLAB in the repository folder.
2. Open one of the following files in the MATLAB editor:
   - `codes/Version_1_C_with_low_aff_dimension.m`
   - `codes/Version_2_structured_full_dimension_C.m`
3. Click **Run**.

Each script is self-contained and saves figures plus a summary `.mat` file under `codes/results/`.

---

## Numerical Experiment

Each experiment produces two main figures:

1. **Test SPO risk**
   - compares full-dimensional SPO+ against reduced SPO+;
   - plotted against the number of labeled training samples.

2. **Learned dimension**
   - tracks the learned dimension `dim(W_hat)` as the labeled sample size increases.

The generated result files are:

### Experiment 1: Low-affine-dimension setting
- `codes/results/low_affdim_spo_risk.png`
- `codes/results/low_affdim_dimW.png`
- `codes/results/low_affdim_summary.mat`

### Experiment 2: Full-dimensional structured setting
- `codes/results/full_dim_corridor_spo_risk.png`
- `codes/results/full_dim_corridor_dimW.png`
- `codes/results/full_dim_corridor_summary.mat`

---

## Sample Result

The image paths below assume that the generated figures are kept under `codes/results/` and committed to the repository.

### Experiment 1: Low-affine-dimension setting

<p align="center">
  <img src="codes/results/low_affdim_spo_risk.png" alt="Low-affine-dimension SPO risk" width="48%">
  <img src="codes/results/low_affdim_dimW.png" alt="Low-affine-dimension learned dimension" width="48%">
</p>

<p align="center">
  Left: test SPO risk. Right: learned dimension of the decision-sufficient subspace.
</p>

### Experiment 2: Full-dimensional structured setting

<p align="center">
  <img src="codes/results/full_dim_corridor_spo_risk.png" alt="Full-dimensional structured SPO risk" width="48%">
  <img src="codes/results/full_dim_corridor_dimW.png" alt="Full-dimensional structured learned dimension" width="48%">
</p>

<p align="center">
  Left: test SPO risk. Right: learned dimension of the decision-sufficient subspace.
</p>

Typical qualitative behavior:
- the learned dimension stabilizes quickly;
- the reduced SPO+ model becomes competitive with fewer labeled samples;
- the compression effect is clearest in the full-dimensional structured experiment.

---

## Notes

- `Version_1_C_with_low_aff_dimension.m` is the simpler warm-up experiment.
- `Version_2_structured_full_dimension_C.m` is the main compression experiment.
- This repository is intended as a compact numerical illustration of decision-sufficient representation learning for contextual linear optimization.

---

## Author

Yuhan Ye
