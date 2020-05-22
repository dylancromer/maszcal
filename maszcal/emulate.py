from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
import sklearn.gaussian_process.kernels
import pality
import maszcal.mathutils


class LensingPca:
    @classmethod
    def create(cls, lensing_data):
        return cls.get_pca(cls.standardize(lensing_data))

    @classmethod
    def subtract_mean(cls, array):
        return array - array.mean(axis=-1)[:, None]

    @classmethod
    def normalize_by_std(cls, array):
        return array / array.std(axis=-1)[:, None]

    @classmethod
    def standardize(cls, lensing_data):
        shifted_data = cls.subtract_mean(lensing_data)
        scaled_shifted_data = cls.normalize_by_std(shifted_data)
        return scaled_shifted_data

    @classmethod
    def get_pca(cls, data):
        return pality.Pca.calculate(data)


@dataclass
class PcaEmulator:
    mean: np.ndarray
    std_dev: np.ndarray
    coords: np.ndarray
    basis_vectors: np.ndarray
    weights: np.ndarray
    interpolator_class: object
    interpolator_kwargs: object = MappingProxyType({})

    def __post_init__(self):
        self.n_components = self.basis_vectors.shape[-1]
        self.weight_interpolators = self.create_weight_interpolators()

    def create_weight_interpolators(self):
            return tuple(
                self.interpolator_class(self.coords, self.weights[i, :], **self.interpolator_kwargs) for i in range(self.n_components)
            )

    def reconstruct_standard_data(self, coords):
        return np.stack(tuple(
            self.weight_interpolators[i](coords)[None, :] * self.basis_vectors[:, i, None] for i in range(self.n_components)
        )).sum(axis=0)

    def reconstruct_data(self, coords):
        standard_data = self.reconstruct_standard_data(coords)
        return self.mean[:, None] + (standard_data*self.std_dev[:, None])

    def __call__(self, coords):
        return self.reconstruct_data(coords)

    @classmethod
    def create_from_data(cls, coords, data, interpolator_class, interpolator_kwargs=MappingProxyType({}), num_components=8):
        pca = LensingPca.create(data)
        basis_vectors = pca.basis_vectors[:, :num_components]
        weights = pca.weights[:num_components, :]
        return cls(
            mean=data.mean(axis=-1),
            std_dev=data.std(axis=-1),
            coords=coords,
            basis_vectors=basis_vectors,
            weights=weights,
            interpolator_class=interpolator_class,
            interpolator_kwargs=interpolator_kwargs,
        )