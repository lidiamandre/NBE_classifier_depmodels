# NBEs and model selection classifiers

This repository contains all the neural Bayes estimators (NBEs) and model selection classifiers trained for the paper "Neural Bayes estimation for complex bivariate extremal dependence models". 

All the estimators were trained for sample sizes $n\in (100, 1500)$ and, in the case of censored data, with censoring levels $\tau \in (0.55, 0.99).$ For Model W and the WCM with Model E1 as the tail component, the neural interval estimators are givne in the folder "Interval Estimator".

For the model selection classifiers, the class index is as follows:
  - Binary classification: $\zeta = 1$ for the model that first appears in the respective folder. For example, for folder "Models E1 and E2", $\zeta = 1$ refers to Model E1 and $\zeta = 2$ refers to Model E2
  - Multiclass classification: $\zeta = 1$ refers to Model W, $\zeta = 2$ to Model HW, $\zeta = 3$ to Model E1, and $\zeta = 4$ to Model E2.

The instructions how to load the estimators in [`Julia`](https://julialang.org/downloads/) are given in file `Loading.jl`.


