import hybrid_flows.distributions as _hf_d

from dcp_nf_forecast.distributions import (
    get_multivariate_lognormal_diag,
    get_multivariate_normal_diag,
)

_hf_d.get_multivariate_normal_diag = get_multivariate_normal_diag
_hf_d.get_multivariate_lognormal_diag = get_multivariate_lognormal_diag
