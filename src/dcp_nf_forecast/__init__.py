import hybrid_flows.distributions as _hf_d

from dcp_nf_forecast.distributions import (
    get_multivariate_lognormal,
)

_hf_d.get_multivariate_lognormal = get_multivariate_lognormal
