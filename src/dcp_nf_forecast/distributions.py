from collections.abc import Callable
from functools import partial

import hybrid_flows.distributions as hf_distributions
import hybrid_flows.parameters as hf_parameters
import tensorflow as tf
import tensorflow_probability as tfp


def _get_multivariate_normal_diag_fn(
    dims: int,
) -> tuple[Callable[[tf.Tensor], tfp.distributions.Distribution], tuple[int, ...]]:
    parameters_shape = (2 * dims,)

    def dist(parameters: tf.Tensor) -> tfp.distributions.Distribution:
        loc = parameters[..., :dims]
        scale = tf.math.softplus(parameters[..., dims:]) + 1e-6
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    return dist, parameters_shape


def _get_multivariate_lognormal_diag_fn(
    dims: int,
) -> tuple[Callable[[tf.Tensor], tfp.distributions.Distribution], tuple[int, ...]]:
    parameters_shape = (2 * dims,)

    def dist(parameters: tf.Tensor) -> tfp.distributions.Distribution:
        loc = parameters[..., :dims]
        scale = tf.math.softplus(parameters[..., dims:]) + 1e-6
        mv_normal = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        return tfp.distributions.TransformedDistribution(mv_normal, tfp.bijectors.Exp())

    return dist, parameters_shape


get_multivariate_normal_diag = partial(
    hf_distributions._get_trainable_distribution,
    get_distribution_fn=_get_multivariate_normal_diag_fn,
    parameters_fn=hf_parameters.get_parameter_vector_or_simple_network_fn,
)

get_multivariate_lognormal_diag = partial(
    hf_distributions._get_trainable_distribution,
    get_distribution_fn=_get_multivariate_lognormal_diag_fn,
    parameters_fn=hf_parameters.get_parameter_vector_or_simple_network_fn,
)
