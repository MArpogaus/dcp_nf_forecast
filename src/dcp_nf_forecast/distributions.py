"""Custom probability distributions for normalizing flows."""

from collections.abc import Callable
from functools import partial

import hybrid_flows.distributions as hf_distributions
import hybrid_flows.parameters as hf_parameters
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def _get_multivariate_lognormal_fn(
    dims: int,
) -> tuple[Callable[[tf.Tensor], tfp.distributions.Distribution], tuple[int, ...]]:

    mv_normal_dist, parameters_shape = hf_distributions._get_multivariate_normal_fn(
        dims
    )

    def dist(parameters: tf.Tensor) -> tfd.Distribution:
        mv_normal = mv_normal_dist(parameters)
        return tfp.distributions.TransformedDistribution(mv_normal, tfp.bijectors.Exp())

    return dist, parameters_shape


get_multivariate_lognormal = partial(
    hf_distributions._get_trainable_distribution,
    get_distribution_fn=_get_multivariate_lognormal_fn,
    parameters_fn=hf_parameters.get_parameter_vector_or_simple_network_fn,
)


def _get_multivariate_truncated_normal_fn(
    dims: int,
) -> tuple[Callable[[tf.Tensor], tfp.distributions.Distribution], tuple[int, ...]]:

    mv_normal_dist, parameters_shape = hf_distributions._get_multivariate_normal_fn(
        dims
    )

    def dist(parameters: tf.Tensor) -> tfd.Distribution:
        mv_normal = mv_normal_dist(parameters)
        loc = mv_normal.mean()
        scale = mv_normal.stddev()
        return tfd.Independent(
            tfd.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=5.0),
            reinterpreted_batch_ndims=1,
        )

    return dist, parameters_shape


get_multivariate_truncated_normal = partial(
    hf_distributions._get_trainable_distribution,
    get_distribution_fn=_get_multivariate_truncated_normal_fn,
    parameters_fn=hf_parameters.get_parameter_vector_or_simple_network_fn,
)
