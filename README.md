# Learning Decision-Sufficient Representations for Linear Optimization

This repository contains compact MATLAB experiments accompanying the paper **Learning Decision-Sufficient Representations for Linear Optimization** by **Yuhan Ye, Saurabh Amin, and Asuman Ozdaglar**.

The code is designed for readers who want a concrete, self-contained shortest-path illustration of the paper's central message: **if optimal decisions depend only on a low-dimensional decision-relevant subspace, then contextual predictors can be compressed to that subspace and can generalize more efficiently than full-dimensional predictors**.

## Relation to the Paper

This repository is most closely connected to the following parts of the paper:

- **Sections 3-6**: decision-sufficient datasets, the decision-relevant subspace `W*`, pointwise sufficiency, and cumulative learning over samples.
- **Section 7**: model compression for contextual linear optimization (CLO).
- **Section 7.4**: the synthetic shortest-path numerical illustration.

At the same time, this repository should be read as a **reader-facing reference implementation**, not as a full reproduction package for every theorem, appendix, or experiment in the manuscript.

In particular:

- The repository **does implement** synthetic shortest-path experiments that compare full-dimensional SPO+ training against reduced SPO+ training after learning a low-dimensional decision-relevant subspace.
- The repository **does not implement** every algorithmic and theoretical component from the paper in full generality.
- The repository **does not aim to reproduce every quantitative number** in the manuscript line-by-line.
- The included paper **Active Learning For Contextual Linear Optimization: A Margin-Based Approach** is relevant background for CLO, SPO+, and shortest-path examples, but its margin-based active learning algorithm is **not** reimplemented here.

## Paper in Brief

The paper studies linear optimization problems of the form

```text
min_x c^T x   subject to x in X,
```

where the feasible set `X` is known but the cost vector `c` is unknown and only known to lie in a prior set `C`.

Instead of observing the full cost vector, one can query linear measurements `q^T c`. The guiding question is:

> Which measurements, and how many of them, are sufficient to recover an optimal decision?

A central object in the paper is the **decision-relevant subspace** `W*`, whose dimension `d*` measures the intrinsic number of directions that can actually change the optimizer. For open convex priors, global sufficiency is characterized by whether the queried directions span this subspace.

The paper then makes three main conceptual moves:

1. It shows that computing globally minimal sufficient datasets is hard in the worst case.
2. It introduces **pointwise sufficiency** as a tractable per-instance relaxation and develops cutting-plane style learning procedures.
3. It applies the learned representation to **contextual linear optimization**, where predictor training can be carried out in a compressed subspace of dimension `d*` instead of the ambient cost dimension `d`.

The theoretical payoff is that the relevant complexity term in contextual optimization can scale with `d*` rather than `d`.

## What This Code Is Meant to Illustrate

The MATLAB code focuses on one qualitative message from the paper:

1. **Stage I:** learn a low-dimensional subspace `W_hat` that captures the decision-relevant directions.
2. **Stage II:** train a reduced contextual predictor inside `W_hat` instead of directly in the full ambient space.
3. Compare the reduced model against a standard full-dimensional SPO+ baseline.

Both experiments in this repository are finite shortest-path toy models. They deliberately use a compact implementation so that the code is easy to read and modify.

Because of that design choice, the code is **simpler than the full formal setup in Section 7**. For example, the paper's CLO discussion uses an ellipsoidal prior and a canonical lifting map, whereas the code here uses simpler shortest-path constructions with box-style restrictions and direct finite-path computations. The goal is not to mirror every formal object exactly, but to preserve the same **model-compression story** in a transparent numerical setting.

## Experiments Included in This Repository

### 1. Low-Affine-Dimension Box Experiment

Public entry point:

- `run_low_affdim_box_experiment.m`

Implementation file:

- `initial_C_low_aff_dimension.m`

What it does:

- Builds a monotone shortest-path problem.
- Restricts the cost set so that only a small subset of edge coordinates can vary.
- Learns decision-relevant directions online from labeled cost samples.
- Compares a full-dimensional SPO+ predictor against a reduced predictor that only operates inside the learned subspace.

Why it is useful:

- This is the cleanest warm-up experiment in the repository.
- It makes the compression effect easy to see because the intrinsic structure is intentionally simple.
- It is **not** the strongest setting from the paper, because here the prior itself already has low affine complexity.

### 2. Full-Dimensional Corridor Experiment

Public entry point:

- `run_full_dim_corridor_experiment.m`

Implementation file:

- `structured_full_dimension_C_with_v1_plot.m`

What it does:

- Builds a shortest-path problem in which the prior cost set is still full-dimensional.
- Uses a corridor construction so that only a small family of paths can become optimal.
- Learns a decision-relevant subspace from observed costs.
- Compares full-dimensional SPO+ against reduced SPO+ in the learned subspace.

Why it is closer to the paper:

- The paper's main representation-learning story is interesting precisely when the ambient cost space is high-dimensional but the optimizer only depends on a much smaller intrinsic structure.
- This corridor experiment is a compact version of that phenomenon.
- Among the scripts in this repository, this is the one that is conceptually closest to the Section 7.4 shortest-path illustration.

## Repository Layout

```text
run_low_affdim_box_experiment.m         Public entry point for the low-affine-dimension experiment
run_full_dim_corridor_experiment.m      Public entry point for the full-dimensional corridor experiment
run_all_experiments.m                   Runs both experiments

initial_C_low_aff_dimension.m           Implementation of the low-affine-dimension experiment
structured_full_dimension_C_with_v1_plot.m
                                        Implementation of the full-dimensional corridor experiment

README.md                               This document
results/                                Created automatically when experiments are run
```

Legacy filenames are kept for backward compatibility, but new readers should start from the `run_*` entry points.

## Requirements

- MATLAB R2021a or later
- Optimization Toolbox (`linprog`)

No external datasets are required.

## Quick Start

Run either experiment individually from MATLAB:

```matlab
run_low_affdim_box_experiment
run_full_dim_corridor_experiment
```

Or run both:

```matlab
run_all_experiments
```

## What Gets Saved

Each experiment automatically creates a `results/` directory and saves:

- exported figures as `.png`
- a summary `.mat` file containing the experiment configuration and aggregated metrics

Typical outputs are:

- `results/low_affdim_spo_risk.png`
- `results/low_affdim_dimW.png`
- `results/low_affdim_summary.mat`
- `results/full_dim_corridor_spo_risk.png`
- `results/full_dim_corridor_dimW.png`
- `results/full_dim_corridor_summary.mat`

## How to Read the Outputs

Each experiment produces two main plots.

### Learned Dimension Plot

This plot tracks the dimension of the learned subspace `W_hat` as more labeled samples are observed.

Interpretation:

- If the learned dimension quickly stabilizes, Stage I is discovering the relevant directions efficiently.
- If it stabilizes near the expected intrinsic dimension, the representation learner is behaving as intended.

### SPO Risk Plot

This plot compares test SPO risk for:

- a full-dimensional predictor
- a reduced predictor trained after learning `W_hat`

Interpretation:

- If the reduced model reaches lower risk earlier, the learned representation is helping statistically.
- This is the qualitative phenomenon predicted by the paper's model-compression perspective: learning in the right low-dimensional subspace can be better than learning in the full ambient space.

## Mapping from Paper to Code

The repository is easiest to understand through the following paper-to-code map.

### From the paper

- The paper defines the decision-relevant subspace `W*` and intrinsic dimension `d*`.
- It shows that worst-case global sufficiency is computationally hard.
- It introduces pointwise sufficiency and cumulative learning procedures.
- It applies the resulting representation to contextual linear optimization and shows that the effective dimension in the generalization bound can improve from `d` to `d*`.

### In the code

- `W_hat` plays the role of a learned approximation to the decision-relevant subspace.
- The online/cumulative updates are simplified shortest-path versions of the representation-learning idea.
- The reduced predictor corresponds to Stage II compression: train only in the learned subspace instead of the full cost dimension.
- The SPO risk comparison is the practical visualization of the paper's Section 7 message.

## Important Scope Notes

To avoid overstating what this repository does, here are the main limitations.

- This is a **synthetic shortest-path repository**, not a full benchmark suite.
- The code is meant to be **readable and minimal**, so several objects from the theory are implemented in simplified form.
- The repository gives a **qualitative numerical illustration** of the paper's compression idea; it is not a theorem-verification package.
- The full ellipsoidal-prior formulation and all appendix-level technical details are not implemented verbatim.
- Existing PNG files in the repository root are older snapshots; new runs save outputs under `results/`.

## Main User Controls

If you want to modify the experiments, the most useful parameters are near the top of the implementation files.

For `initial_C_low_aff_dimension.m`:

- `cfg.r_true`: target intrinsic dimension in the warm-up experiment
- `cfg.Ntrain`, `cfg.Ntest`, `cfg.nTrials`: dataset sizes and repetition count
- `cfg.seed`: global reproducibility seed

For `structured_full_dimension_C_with_v1_plot.m`:

- `cfg.dstar_target`: target intrinsic dimension for the corridor construction
- `cfg.trainSizes`: training sample sizes used in the performance curve
- `cfg.nTrial`: number of random trials
- `cfg.seed`: global reproducibility seed

## Related Reading in This Folder

- Learning_Decision_Sufficient_Representations_for_Linear_Optimization_arxiv.pdf: the main paper motivating this repository.
- Active Learning For Contextual Linear Optimization A Margin-Base Approach.pdf: relevant background on CLO, SPO+, shortest-path examples, and label-efficient learning.

## Citation
If you use or build on this repository, please cite the paper:

```bibtex
@misc{ye2026decisionsufficient,
  title={Learning Decision-Sufficient Representations for Linear Optimization},
  author={Yuhan Ye and Saurabh Amin and Asuman Ozdaglar},
  year={2026},
  note={Preprint}
}
```

If the paper later appears with an official arXiv identifier, conference version, or journal version, please update the citation accordingly.



