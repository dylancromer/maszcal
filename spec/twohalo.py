from dataclasses import dataclass
import pytest
import numpy as np
import maszcal.twohalo
import maszcal.cosmology
import maszcal.interp_utils


@dataclass
class FakePower:
    cosmo_params: object

    def get_spectrum_interpolator(self, ks, zs, is_nonlinear):
        return lambda ks, zs: np.ones(zs.shape + ks.shape)


def fake_quad_vec(func, a, b, limit=1):
    return np.ones_like(func(a)), 1


@dataclass
class FakePowerInterpolator:
    nonlinear: bool

    def P(self, zs, ks):
        return (ks**4)[None, :] * np.ones(zs.shape + ks.shape)


@dataclass
class FakeCambResults:
    nonlinear: bool

    def calc_power_spectra(self):
        pass

    def get_matter_power_interpolator(self, **kwargs):
        return FakePowerInterpolator(nonlinear=self.nonlinear)


def fake_camb_get_results(params):
    if params.NonLinear == 'NonLinear_none':
        return FakeCambResults(nonlinear=False)
    else:
        return FakeCambResults(nonlinear=True)


def describe_TwoHaloConvergenceModel():

    def describe_angle_scale_distance():

        @pytest.fixture
        def model_comoving(mocker):
            mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(
                cosmo_params=cosmo,
                matter_power_class=FakePower,
            )
            model.NUM_INTERP_ZS = 3
            model.NUM_INTERP_RADII = 4
            return model

        @pytest.fixture
        def model_physical(mocker):
            mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(
                cosmo_params=cosmo,
                matter_power_class=FakePower,
            )
            model.NUM_INTERP_ZS = 3
            model.NUM_INTERP_RADII = 4
            model.COMOVING = False
            return model

        def it_differs_between_comoving_and_noncomoving_cases(model_physical, model_comoving):
            zs = np.random.rand(4) + 0.1
            scale_physical = model_physical.angle_scale_distance(zs)
            scale_comoving = model_comoving.angle_scale_distance(zs)
            assert np.all(scale_physical != scale_comoving)


    def describe_convergence():

        @pytest.fixture
        def two_halo_model(mocker):
            mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(
                cosmo_params=cosmo,
                matter_power_class=FakePower,
                sd_kwargs={'radial_axis_to_broadcast': 1, 'density_axis': -1},
            )

            model.NUM_INTERP_ZS = 3
            model.NUM_INTERP_RADII = 4

            return model

        def it_calculates_two_halo_convergences(two_halo_model):
            zs = np.linspace(0.1, 1, 4)
            mus = np.linspace(32, 33, 4)
            from_arcmin = 2 * np.pi / 360 / 60
            thetas = np.logspace(-4, np.log10(15*from_arcmin), 2)

            sds = two_halo_model.convergence(thetas, zs, mus)

            assert not np.any(np.isnan(sds))
            assert sds.shape == zs.shape + thetas.shape


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model(mocker):
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo, matter_power_class=FakePower)

        model.NUM_INTERP_ZS = 3
        model.NUM_INTERP_RADII = 4

        return model

    def it_calculates_two_halo_excess_surface_densities(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.linspace(32, 33, 4)
        rs = np.logspace(-1, 1, 2)

        excess_surface_densities = two_halo_model.excess_surface_density(rs, zs, mus)

        assert not np.any(np.isnan(excess_surface_densities))
        assert excess_surface_densities.shape == zs.shape + rs.shape


def describe_TwoHaloModel():

    @pytest.fixture
    def two_halo_model(mocker):
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloModel(cosmo_params=cosmo, matter_power_class=FakePower)

        model.NUM_INTERP_ZS = 3
        model.NUM_INTERP_RADII = 4

        return model

    def it_calculates_halo_matter_correlations(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.linspace(32, 33, 4)
        rs = np.logspace(-1, 1, 2)

        xis = two_halo_model.halo_matter_correlation(rs, zs, mus)

        assert not np.any(np.isnan(xis))
        assert xis.shape == zs.shape + rs.shape

    def it_reshapes_the_correlator_correctly(two_halo_model):
        zs = np.linspace(0, 1, 4)
        rs_1 = np.logspace(-1, 1, 5)
        rs_2 = np.logspace(-1.1, 0.9, 5)
        rs_3 = np.logspace(-0.9, 1.1, 5)
        rs = np.stack((rs_1, rs_2, rs_3))

        xis_1 = two_halo_model._density_shape_interpolator(rs, zs)
        xis_2 = np.array([two_halo_model._density_shape_interpolator(r, zs) for r in rs])
        assert np.all(xis_1 == xis_2)


def describe_TwoHaloEmulator():

    def describe_context_esd():

        @pytest.fixture
        def two_halo_esd(mocker):
            mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo, matter_power_class=FakePower)
            model.NUM_INTERP_ZS = 3
            model.NUM_INTERP_RADII = 4
            return model.excess_surface_density

        @pytest.fixture
        def esd_emulator(two_halo_esd):
            return maszcal.twohalo.TwoHaloEmulator.from_function(
                two_halo_func=two_halo_esd,
                r_grid=np.geomspace(0.1, 10, 13),
                z_lims=np.array([0, 2]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=10,
            )

        def it_can_calculate_esds(esd_emulator):
            rs = np.geomspace(1, 3, 30)

            rng = np.random.default_rng(seed=13)
            NUM = 10
            mus = np.log(2e13*rng.normal(size=NUM) + 2e14)
            zs = rng.random(size=NUM) + 0.01

            esds = esd_emulator(rs, zs, mus)
            assert not np.any(np.isnan(esds))
            assert esds.shape == zs.shape + rs.shape

        def describe_non_matching_version():

            @pytest.fixture
            def two_halo_esd(mocker):
                mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
                cosmo = maszcal.cosmology.CosmoParams()
                model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo, matter_power_class=FakePower)
                model.NUM_INTERP_ZS = 3
                model.NUM_INTERP_RADII = 4
                return model.excess_surface_density

            @pytest.fixture
            def esd_emulator(two_halo_esd):
                return maszcal.twohalo.TwoHaloEmulator.from_function(
                    two_halo_func=two_halo_esd,
                    r_grid=np.geomspace(0.1, 10, 13),
                    z_lims=np.array([0, 2]),
                    mu_lims=np.log(np.array([1e13, 1e15])),
                    num_emulator_samples=10,
                    separate_mu_and_z_axes=True,
                )

            def it_can_calculate_esds(esd_emulator):
                rs = np.geomspace(1, 3, 30)

                rng = np.random.default_rng(seed=13)
                mus = np.log(2e13*rng.normal(size=9) + 2e14)
                zs = rng.random(size=10) + 0.01

                esds = esd_emulator(rs, zs, mus)
                assert not np.any(np.isnan(esds))
                assert esds.shape == mus.shape + zs.shape + rs.shape

    def describe_context_convergence():

        @pytest.fixture
        def two_halo_conv(mocker):
            mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(cosmo_params=cosmo, matter_power_class=FakePower)
            model.NUM_INTERP_ZS = 5
            model.NUM_INTERP_RADII = 7
            return model.radius_space_convergence

        @pytest.fixture
        def conv_emulator(two_halo_conv):
            return maszcal.twohalo.TwoHaloEmulator.from_function(
                two_halo_func=two_halo_conv,
                r_grid=np.geomspace(0.1, 10, 13),
                z_lims=np.array([0, 2]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=10,
            )

        def it_can_calculate_convergence(conv_emulator):
            rs = np.geomspace(1, 3, 30)

            rng = np.random.default_rng(seed=13)
            NUM = 10
            mus = np.log(2e13*rng.normal(size=NUM) + 2e14)
            zs = rng.random(size=NUM) + 0.01

            convs = conv_emulator(rs, zs, mus)
            assert not np.any(np.isnan(convs))
            assert convs.shape == zs.shape + rs.shape

        def it_can_calculate_convergence_redshift_dep_radii(conv_emulator):
            rng = np.random.default_rng(seed=13)
            NUM = 10
            mus = np.log(2e13*rng.normal(size=NUM) + 2e14)
            zs = rng.random(size=NUM) + 0.01
            rs = np.geomspace(zs+1, 3, 30)

            convs = conv_emulator.with_redshift_dependent_radii(rs, zs, mus)
            assert not np.any(np.isnan(convs))
            assert convs.shape == zs.shape + rs.shape[0:1]

        def describe_non_matching_version():

            @pytest.fixture
            def two_halo_conv(mocker):
                mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
                cosmo = maszcal.cosmology.CosmoParams()
                model = maszcal.twohalo.TwoHaloConvergenceModel(cosmo_params=cosmo, matter_power_class=FakePower)
                model.NUM_INTERP_ZS = 5
                model.NUM_INTERP_RADII = 7
                return model.radius_space_convergence

            @pytest.fixture
            def conv_emulator(two_halo_conv):
                return maszcal.twohalo.TwoHaloEmulator.from_function(
                    two_halo_func=two_halo_conv,
                    r_grid=np.geomspace(0.1, 10, 13),
                    z_lims=np.array([0, 2]),
                    mu_lims=np.log(np.array([1e13, 1e15])),
                    num_emulator_samples=10,
                    separate_mu_and_z_axes=True,
                )

            def it_can_calculate_convergence(conv_emulator):
                rs = np.geomspace(1, 3, 30)
                rng = np.random.default_rng(seed=13)
                mus = np.log(2e13*rng.normal(size=9) + 2e14)
                zs = rng.random(size=10) + 0.01

                convs = conv_emulator(rs, zs, mus)
                assert not np.any(np.isnan(convs))
                assert convs.shape == mus.shape + zs.shape + rs.shape

            def it_can_calculate_convergence_redshift_dep_radii(conv_emulator):
                rng = np.random.default_rng(seed=13)
                mus = np.log(2e13*rng.normal(size=9) + 2e14)
                zs = rng.random(size=10) + 0.01
                rs = np.geomspace(zs+1, 3, 30)

                convs = conv_emulator.with_redshift_dependent_radii(rs, zs, mus)
                assert not np.any(np.isnan(convs))
                assert convs.shape == mus.shape + zs.shape + rs.shape[0:1]
