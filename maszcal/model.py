from dataclasses import dataclass
import numpy as np
import astropy.units as u
import maszcal.defaults
import maszcal.likelihoods
import maszcal.fitutils


@dataclass
class SingleMass:
    lensing_signal_class: object
    cm_relation: bool
    log_likelihood_func: object = maszcal.likelihoods.log_gaussian_shape
    minimize_func: object = maszcal.fitutils.global_minimize
    minimize_method: str = 'global-differential-evolution'

    def __post_init__(self):
        self._init_lensing_signal_class()

    def _init_lensing_signal_class(self):
        #XXX XXX XXX need to un-hard-code these defaults somehow
        self.lensing_signal_model = self.lensing_signal_class(
            redshift=np.zeros(1),
            units=u.Msun/u.pc**2,
            comoving=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=maszcal.defaults.DefaultCosmology(),
        )
        #XXX XXX XXX

    def esd(self, rs, params):
        if self.cm_relation:
            params = np.concatenate([np.array([3]), params])

        return self.lensing_signal_model.esd(rs, params[None, :])

    def log_likelihood(self, params, data):
        prediction = self.esd(data.radii, params).flatten()
        return self.log_likelihood_func(prediction, data.wl_signals.flatten(), data.covariances)

    def get_best_fit(self, data, param_mins, param_maxes):
        return self.minimize_func(
            lambda params: -self.log_likelihood(params, data),
            param_mins,
            param_maxes,
            self.minimize_method,
        )
