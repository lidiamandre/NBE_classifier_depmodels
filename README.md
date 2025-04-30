# NBEs and NBCs

This repository contains all the neural Bayes estimators (NBEs) and neural Bayes classifiers (NBCs) trained for the paper ["Neural Bayes estimation for complex bivariate extremal dependence models"](https://arxiv.org/abs/2503.23156). 

All the estimators were trained for sample sizes $n\in (100, 1500)$ and, in the case of censored data, with censoring levels $\tau \in (0.55, 0.99).$ For Model W and the WCM with Model E1 as the tail component, the neural interval estimators are given in the folder "Interval Estimator".

For the NBCs, the class index is as follows:
  - $K = 2$: $m = 1$ for the model that first appears in the respective folder. For example, for folder "Models E1 and E2", $m = 1$ refers to Model E1 and $m = 2$ refers to Model E2
  - $K = 4$: $m = 1$ refers to Model W, $m = 2$ to Model HW, $m = 3$ to Model E1, and $m = 4$ to Model E2.

The instructions how to load the estimators in [`Julia`](https://julialang.org/downloads/) are given in file `Loading.jl`.


