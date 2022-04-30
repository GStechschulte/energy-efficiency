import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, PeriodicKernel, RBFKernel, RQKernel


def entsorgung_kernel(time) -> int:
    """
    10 minute aggregation kernels
    """

    period_constraint_short = gpytorch.constraints.Interval(0.08, 0.11)

    seasonal_periodic_short = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_short
        )
        )

    seasonal_rbf_short = ScaleKernel(
        RBFKernel()
        )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21)

    seasonal_periodic_long = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_long
        )
        )

    seasonal_rbf_long = ScaleKernel(
        RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = ScaleKernel(
        RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    if time == 10:
        covar_module = locally_long_periodic + local_variation
        return covar_module

    elif time == 30:
        covar_module = locally_long_periodic + local_variation
        return covar_module

    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def hauptluftung_kernel(time) -> int:
    """
    10 and 30 minute aggregation kernels
    """

    period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21)

    seasonal_periodic_long = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_long
        ))

    seasonal_rbf_long = ScaleKernel(
        RQKernel()
        )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = ScaleKernel(
        RQKernel()
        )

    local_variation.alpha = local_variation_alpha
    
    
    if time == 10:
        covar_module = locally_long_periodic + local_variation

        return covar_module
    
    elif time == 30:
        covar_module = locally_long_periodic + local_variation

        return covar_module
    
    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def gesamtmessung_kernel(time) -> int:
    """
    10 and 30 minute aggregation kernels
    """

    if time == 30:
        period_constraint_short = gpytorch.constraints.Interval(0.06, 0.10)

        seasonal_periodic_short = ScaleKernel(
            PeriodicKernel(
                period_length_constraint=period_constraint_short
            ))

        seasonal_rbf_short = ScaleKernel(
            RBFKernel()
            )

        locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

        period_constraint_long = gpytorch.constraints.Interval(0.20, 0.23)

        seasonal_periodic_long = ScaleKernel(
            PeriodicKernel(
                period_length_constraint=period_constraint_long
            ))

        seasonal_rbf_long = ScaleKernel(
            RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = ScaleKernel(
            RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = (
            locally_short_periodic + locally_long_periodic + local_variation)

        return covar_module
    
    elif time == 10:
        period_constraint_short = gpytorch.constraints.Interval(0.07, 0.10)

        seasonal_periodic_short = ScaleKernel(
            PeriodicKernel(
                period_length_constraint=period_constraint_short
            ))

        seasonal_rbf_short = ScaleKernel(
            RBFKernel()
            )

        locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

        period_constraint_long = gpytorch.constraints.Interval(0.19, 0.21)

        seasonal_periodic_long = ScaleKernel(
            PeriodicKernel(
                period_length_constraint=period_constraint_long
            ))

        seasonal_rbf_long = ScaleKernel(
            RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = ScaleKernel(
            RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = (
            locally_short_periodic + locally_long_periodic + local_variation)

        return covar_module
    
    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def eg_kernel(time) -> int:


    if time == 10:
        period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21)

        seasonal_periodic_long = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_long
        ))

        seasonal_rbf_long = ScaleKernel(
            RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = ScaleKernel(
            RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = locally_long_periodic + local_variation

        return covar_module

    elif time == 30:
        period_constraint_short = gpytorch.constraints.Interval(0.07, 0.11)

        seasonal_periodic_short = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_short
        ))

        seasonal_rbf_short = ScaleKernel(
            RBFKernel()
            )

        locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

        period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21)

        seasonal_periodic_long = ScaleKernel(
            PeriodicKernel(
                period_length_constraint=period_constraint_long
            ))

        seasonal_rbf_long = ScaleKernel(
            RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = ScaleKernel(
            RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = (
            locally_short_periodic + locally_long_periodic + local_variation)

        return covar_module

    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def uv_eg_kernel(time) -> int:

    period_constraint_short = gpytorch.constraints.Interval(0.07, 0.11)

    seasonal_periodic_short = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_short
        ))

    seasonal_rbf_short = ScaleKernel(
        RBFKernel()
        )

    locally_short_periodic = seasonal_periodic_short * seasonal_rbf_short

    period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21)

    seasonal_periodic_long = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_long
        ))

    seasonal_rbf_long = ScaleKernel(
        RBFKernel()
    )

    locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

    local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

    local_variation = ScaleKernel(
        RQKernel()
        )
        
    local_variation.alpha = local_variation_alpha

    if time == 30:
        covar_module = (
            locally_short_periodic + locally_long_periodic + local_variation)
        return covar_module

    elif time == 10:
        covar_module = (
            locally_short_periodic + locally_long_periodic + local_variation)
        return covar_module
    
    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def og_kernel(time) -> int:


    if time == 30:
        period_constraint_long = gpytorch.constraints.Interval(0.18, 0.21)

        seasonal_periodic_long = ScaleKernel(
        PeriodicKernel(
            period_length_constraint=period_constraint_long
        ))

        seasonal_rbf_long = ScaleKernel(
            RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = ScaleKernel(
            RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = locally_long_periodic + local_variation

        return covar_module
    
    elif time == 10:
        period_constraint_long = gpytorch.constraints.Interval(0.19, 0.21)

        seasonal_periodic_long = ScaleKernel(
            PeriodicKernel(
                period_length_constraint=period_constraint_long
            ))

        seasonal_rbf_long = ScaleKernel(
            RBFKernel()
        )

        locally_long_periodic = seasonal_periodic_long * seasonal_rbf_long

        local_variation_alpha = gpytorch.priors.GammaPrior(1, 0.5)

        local_variation = ScaleKernel(
            RQKernel()
            )
            
        local_variation.alpha = local_variation_alpha

        covar_module = locally_long_periodic + local_variation

        return covar_module

    else:
        raise ValueError('Enter either 10 or 30 minute aggregation periods')


def main(machine=str, time=int):
    
    if machine == 'entsorgung':
        kernel_function = entsorgung_kernel(time=time)
    elif machine == 'hauptluftung':
        kernel_function = hauptluftung_kernel(time=time)
    elif machine == 'gesamtmessung':
        kernel_function = gesamtmessung_kernel(time=time)
    elif machine == 'eg':
        kernel_function = eg_kernel(time=time)
    elif machine == 'uv_eg':
        kernel_function = uv_eg_kernel(time=time)
    elif machine == 'og':
        kernel_function = og_kernel(time=time)
    else:
        raise ValueError('Enter a valid machine name')

    return kernel_function

if __name__ == '__main__':
    main()