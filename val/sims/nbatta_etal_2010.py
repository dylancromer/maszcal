import pytest
import numpy as np
import pathos.pools as pp
import maszcal.data.sims
import maszcal.analysis


def describe_nbatta_sim():

    def describe_impact_of_baryons():

        @pytest.fixture
        def sim_data():
            return maszcal.data.sims.NBatta2010()

        def using_baryon_reduces_bias(sim_data):
            NUM_THREADS = 12

            nfw_params_shape = (1, sim_data.radii.size, sim_data.redshifts.size)
            nfw_fits = np.zeros(nfw_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                nfw_model = maszcal.analysis.select_model(
                    data=sim_data.select_redshift_index(i),
                    model='nfw',
                    cm_relation=True,
                    emulation=False,
                    stacked=False
                )

                #def _pool_func(data): return nfw_model.get_best_fit(data)
                #pool = pp.ProcessPool(NUM_THREADS)
                #nfw_fits[:, :, i] = np.array(
                #    pool.map(_pool_func, sim_data.wl_signals[:, :, i].T),
                #).T
                #pool.close()
                #pool.join()

            baryon_params_shape = (4, sim_data.radii.size, sim_data.redshifts.size)
            baryon_fits = np.zeros(baryon_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                baryon_model = maszcal.analysis.select_model(
                    data=sim_data.select_redshift_index(i),
                    model='baryon',
                    cm_relation=False,
                    emulation=False,
                    stacked=False,
                )

                #def _pool_func(data): return baryon_model.get_best_fit(data)
                #pool = pp.ProcessPool(NUM_THREADS)
                #baryon_fits[:, :, i] = np.array(
                #    pool.map(_pool_func, sim_data.wl_signals[:, :, i].T),
                #).T
                #pool.close()
                #pool.join()

            nfw_masses = nfw_model.get_masses_from_params(nfw_fits)
            nfw_bias = _calc_bias(nfw_masses, sim_data.masses)

            baryon_masses = baryon_model.get_masses_from_params(baryon_fits)
            baryon_bias = _calc_bias(baryon_masses, sim_data.masses)

            #TODO: make save_data save all possible variable parameters for the fit with the data
            save_data({'nfw-bias_nbatta-2010_single-mass': nfw_bias, 'baryon-bias_nbatta-2010_single-mass': baryon_bias})
            _plot_biases({'NFW only': nfw_bias, 'Baryons': baryon_bias})

            assert baryon_bias.mean() < nfw_bias.mean()
