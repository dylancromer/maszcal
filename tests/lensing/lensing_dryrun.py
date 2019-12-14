import pytest
import numpy as np
import maszcal.lensing as lensing
import maszcal.model as model


def describe_lensing():

    def describe_single_mass_lensing_signal():

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            redshift = 0.4*np.ones(1)
            delta = 500
            mass_def = 'crit'
            lensing_signal = lensing.SingleMassLensingSignal(redshift=redshift, delta=delta, mass_definition=mass_def)

            esd_500c = lensing_signal.esd(rs, np.array([[mu, con]]))

            delta = 200
            kind = 'mean'
            lensing_signal = lensing.SingleMassLensingSignal(redshift=redshift, delta=delta, mass_definition=mass_def)

            esd_200m = lensing_signal.esd(rs, np.array([[mu, con]]))

            assert np.all(esd_200m < esd_500c)

    def describe_stacked_lensing_signal():

        def it_matches_the_stacked_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 2, 5)
            rs = np.logspace(-1, 1, 5)

            con = 3
            a_sz = 0
            params = np.array([[con, a_sz]])

            lensing_signal = lensing.NfwLensingSignal(log_masses=mus, redshifts=redshifts)
            stacked_model = model.NfwShearModel(mu_bins=mus, redshift_bins=redshifts)

            esd_lensing_signal = lensing_signal.stacked_esd(rs, params)
            esd_stacked_model = stacked_model.stacked_delta_sigma(rs, np.array([con]), np.array([a_sz]))

            assert np.all(esd_stacked_model == esd_lensing_signal)

    def describe_stacked_baryon_lensing_signal():

        def it_matches_the_underlying_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 2, 5)
            rs = np.logspace(-1, 1, 5)

            con = 3
            alpha = 1.1
            beta = 4.2
            gamma = 1.3
            a_sz = -0.01
            params = np.array([[con, alpha, beta, gamma, a_sz]])

            lensing_signal = lensing.BaryonLensingSignal(log_masses=mus, redshifts=redshifts)
            baryon_model = model.BaryonShearModel(mu_bins=mus, redshift_bins=redshifts)

            esd_lensing_signal = lensing_signal.stacked_esd(rs, params)
            esd_baryon_model = baryon_model.stacked_delta_sigma(
                rs,
                np.array([con]),
                np.array([alpha]),
                np.array([beta]),
                np.array([gamma]),
                np.array([a_sz]),
            )

            assert np.all(esd_lensing_signal == esd_baryon_model)

    def describe_miyatake_lensing_signal():

        def it_matches_the_underlying_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 2, 5)
            rs = np.logspace(-1, 1, 5)

            a_sz = -0.01
            params = np.array([[a_sz]])

            lensing_signal = lensing.MiyatakeLensingSignal(log_masses=mus, redshifts=redshifts)
            test_model = model.MiyatakeShearModel(mu_bins=mus, redshift_bins=redshifts)

            esd_lensing_signal = lensing_signal.stacked_esd(rs, params)
            esd_test_model = test_model.stacked_delta_sigma(
                rs,
                np.array([a_sz]),
            )

            assert np.all(esd_lensing_signal == esd_test_model)
