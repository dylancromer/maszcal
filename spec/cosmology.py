import pytest
import numpy as np
import maszcal.cosmology as cosmology
from maszcal.cosmology import CosmoParams


def describe_cosmo_params():

    def it_can_pass_checks_with_default_values():
        CosmoParams()

    def it_checks_for_inconsistent_universe_closure():
        with pytest.raises(cosmology.NonFlatUniverseError):
            CosmoParams(
                omega_matter=1,
                omega_lambda=1,
            )

    def it_checks_for_inconsistent_dm_and_bary_densities():
        with pytest.raises(cosmology.MatterInconsistencyError):
            CosmoParams(
                omega_cdm=0.1,
                omega_bary=0.01,
                omega_matter=0.2,
                omega_lambda=0.8,
            )

    def it_checks_for_h_squared_param_consistency():
        with pytest.raises(cosmology.MatterInconsistencyError):
            CosmoParams(omega_cdm_hsqr=0.1)

        with pytest.raises(cosmology.MatterInconsistencyError):
            CosmoParams(omega_bary_hsqr=0.2)

    def it_checks_for_consistence_between_hubble_and_h100():
        with pytest.raises(cosmology.HubbleConstantError):
            CosmoParams(hubble_constant=100)


def describe_SigmaCrit():

    def describe_comoving_case():

        @pytest.fixture
        def sigma_crit():
            return cosmology.SigmaCrit(cosmology.CosmoParams(), comoving=True)

        def it_calculates_sigma_crit(sigma_crit):
            z_sources = np.random.rand(10) + 1100
            z_lenses = np.random.rand(10)

            sd_crit = sigma_crit.sdc(z_sources, z_lenses)
            assert not np.any(np.isnan(sd_crit))
            assert np.all(sd_crit >= 0)

        def it_fails_for_sources_closer_than_lenses(sigma_crit):
            z_sources = np.random.rand(10)
            z_lenses = np.random.rand(10) + 1
            with pytest.raises(ValueError):
                sigma_crit.sdc(z_sources, z_lenses)

    def describe_physical_case():

        @pytest.fixture
        def sigma_crit():
            return cosmology.SigmaCrit(cosmology.CosmoParams(), comoving=False)

        @pytest.fixture
        def sigma_crit_comov():
            return cosmology.SigmaCrit(cosmology.CosmoParams(), comoving=True)

        def it_calculates_sigma_crit(sigma_crit, sigma_crit_comov):
            z_sources = np.random.rand(10) + 1100
            z_lenses = np.random.rand(10)

            sd_crit = sigma_crit.sdc(z_sources, z_lenses)
            assert not np.any(np.isnan(sd_crit))
            assert np.all(sd_crit >= 0)

            sd_crit_comov = sigma_crit_comov.sdc(z_sources, z_lenses)

            assert np.all(sd_crit_comov != sd_crit)
