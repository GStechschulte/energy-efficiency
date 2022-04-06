import torch
import gpytorch


def entsorgung_kernel():
    """
    10 minute aggregation kernels
    """

    ## Locally Periodic Kernels ##
    period_constraint_short = gpytorch.constraints.Interval(0.08, 0.11) ## short term

    seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel(
            period_length_constraint=period_constraint_short
            )
        )

    seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21) ## long term

    seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel(
            period_length_constraint=period_constraint_long
            )
        )

    seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    ## Variations Not Captured By The Trend ##
    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    local_variation_2 = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        )

    covar_module = locally_short_periodic + locally_long_periodic + local_variation

    return covar_module