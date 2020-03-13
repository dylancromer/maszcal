import numpy as np
from maszcal.data.obs import ActHsc2018


def describe_act_hsc_2018():

    def describe_covariance():

        def it_loads_the_covariance_matrix():
            DIR = 'data/act-hsc/'

            radii = np.logspace(0, 1, 10)

            cov = ActHsc2018.covariance(DIR, radii)

            assert isinstance(cov, np.ndarray)
            assert not np.any(np.isnan(cov))
