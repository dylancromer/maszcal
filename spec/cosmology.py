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
