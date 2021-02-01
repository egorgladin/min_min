# min_min
## Solving convex min-min problems with smoothness and strong convexity in one variable group and small dimension of the other
**Dataset:** <a href=https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#madelon>madelon</a>, 2000 objects, 500 features

**Model:** logistic regression with L2-regularization applied to all except *d* model parameters. Regularization coefficient *= 0.05*

**Baseline method:** Varag (<a href=https://arxiv.org/abs/1905.12412>arxiv link</a>)

Results for *d=20*
![alt text](https://github.com/egorgladin/min_min/blob/main/plots/plot_d20.png?raw=true)

Results for *d=30*
![alt text](https://github.com/egorgladin/min_min/blob/main/plots/plot_d30.png?raw=true)
