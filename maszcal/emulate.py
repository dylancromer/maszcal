from dataclasses import dataclass
from types import MappingProxyType
import dill
import numpy as np
import scipy.interpolate
import pality
import ostrich.emulate
from sklearn.gaussian_process.kernels import Matern
import maszcal.interpolate
import maszcal.interp_utils


LensingPca = ostrich.emulate.DataPca

PcaEmulator = ostrich.emulate.PcaEmulator

save_pca_emulator = ostrich.emulate.save_pca_emulator

load_pca_emulator = ostrich.emulate.load_pca_emulator


@dataclass
class LensingFunctionEmulator:
    RADIAL_INTERPOLATION_METHOD = 'cubic'

    radii: np.ndarray
    log_masses: np.ndarray
    redshifts: np.ndarray
    params: np.ndarray
    data: np.ndarray
    interpolator_class: object = maszcal.interpolate.GaussianProcessInterpolator
    interpolator_kwargs: MappingProxyType = MappingProxyType({'kernel': Matern()})
    num_principal_components_base: int = 8
    num_principal_components_2d: int = 8

    def _wrap_base_emulator(self, emulator):
        def wrapper(arg): return emulator(arg).reshape(self.radii.size, self.log_masses.size, self.redshifts.size, -1)
        return wrapper

    def __post_init__(self):
        self._base_emulator = self._wrap_base_emulator(PcaEmulator.create_from_data(
            coords=self.params,
            data=self.data.reshape(-1, self.params.shape[0]),
            interpolator_class=self.interpolator_class,
            interpolator_kwargs=self.interpolator_kwargs,
            num_components=self.num_principal_components_base,
        ))

    def _wrap_2d_emulator(self, emulator):
        def wrapper(mus, zs):
            mus_and_zs = maszcal.interp_utils.cartesian_prod(mus, zs)
            return np.moveaxis(
                emulator(mus_and_zs).reshape(-1, self.radii.size, mus.size, zs.size),
                0,
                -1,
            )
        return wrapper

    def _get_2d_emulator(self, data):
        mus_and_zs = maszcal.interp_utils.cartesian_prod(self.log_masses, self.redshifts)
        data_ = np.moveaxis(data, -1, 0).reshape(-1, mus_and_zs.shape[0])
        return self._wrap_2d_emulator(PcaEmulator.create_from_data(
            coords=mus_and_zs,
            data=data_,
            interpolator_class=self.interpolator_class,
            interpolator_kwargs=self.interpolator_kwargs,
            num_components=self.num_principal_components_2d,
        ))

    def _get_radial_interpolator(self, data):
        return scipy.interpolate.interp1d(
            self.radii,
            data,
            kind=self.RADIAL_INTERPOLATION_METHOD,
            axis=0,
        )

    def __call__(self, radii, log_masses, redshifts, params):
        return self._get_radial_interpolator(
            self._get_2d_emulator(
                self._base_emulator(params),
            )(log_masses, redshifts),
        )(radii)
