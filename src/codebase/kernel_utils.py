import torch
import gpytorch


def entsorgung_kernel(time) -> int:
    """
    10 minute aggregation kernels
    """

    seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    if time == 10:
        covar_module = locally_long_periodic + local_variation

        return covar_module

    elif time == 30:
        covar_module = locally_long_periodic + local_variation

        return covar_module


def hauptluftung_kernel(time) -> int:
    """
    10 and 30 minute aggregation kernels
    """
    if time == 10:
        seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
            )

        seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long
 
        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )
        
        local_variation.alpha = local_variation_alpha

        covar_module = locally_long_periodic + local_variation

        return covar_module
    
    elif time == 30:
        seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
            )

        seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )
        
        local_variation.alpha = local_variation_alpha

        covar_module = locally_long_periodic + local_variation

        return covar_module


def gesamtmessung_kernel(time) -> int:
    """
    10 and 30 minute aggregation kernels
    """

    if time == 30:
        seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
            )

        seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
            )

        locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

        seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
            )

        seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = locally_short_periodic + locally_long_periodic + local_variation

        return covar_module
    
    elif time == 10:
        seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
            )

        seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
            )

        locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

        seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
            )

        seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = locally_short_periodic + locally_long_periodic + local_variation

        return covar_module


def eg_kernel(time) -> int:

    seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    if time == 10:
        covar_module = locally_long_periodic + local_variation

        return covar_module

    elif time == 30:
        covar_module = locally_short_periodic + locally_long_periodic + local_variation

        return covar_module

    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def uv_eg_kernel(time) -> int:

    seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    if time == 30:
        covar_module = locally_short_periodic + locally_long_periodic + local_variation

        return covar_module
    if time == 10:
        covar_module = locally_short_periodic + locally_long_periodic + local_variation

        return covar_module


def og_kernel(time) -> int:

    seasonal_periodic_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_short = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
        )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    seasonal_periodic_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel()
        )

    seasonal_rbf_long = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    if time == 30:
        covar_module = locally_long_periodic + local_variation

        return covar_module
    if time == 10:
        covar_module = locally_long_periodic + local_variation

        return covar_module