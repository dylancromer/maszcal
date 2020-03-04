from dataclasses import dataclass
import numpy as np
import astropy.units as u
import maszcal.defaults
import maszcal.likelihoods
import maszcal.fitutils


@dataclass
class SingleMass:
    lensing_signal_class: object
    log_likelihood_func: object = maszcal.likelihoods.log_gaussian_shape
    optimize_method: str = 'global-differential-evolution'

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
        return self.lensing_signal_model.esd(rs, params)

    def log_likelihood(self, params, data):
        prediction = self.esd(data.radii, params)
        return self.log_likelihood_func(prediction, data.wl_signals, data.covariances)

    def _minimize(self, func_to_minimize, param_mins, param_maxes):
        return maszcal.fitutils.minimize(func_to_minimize, param_mins, param_maxes, method=self.optimize_method)

    def get_best_fit(self, data, param_mins, param_maxes):
        return self._minimize(lambda params: -self.log_likelihood(params, data), param_mins, param_maxes)
