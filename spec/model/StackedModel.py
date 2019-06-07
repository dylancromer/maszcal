import numpy as np
from maszcal.model import StackedModel




def test_mu_sz():
    stacked_model = StackedModel()

    mus = np.linspace(1, 10)
    stacked_model.b_sz = np.linspace(-5,5)

    a_sz = np.ones(1)
    con = None
    radius = None

    stacked_model.set_coords((radius, con, a_sz))

    mu_szs = stacked_model.mu_sz(mus)

    precomp_mu_szs = np.array(
        [-4.        , -4.67680133, -5.2786339 , -5.80549771, -6.25739275,
         -6.63431903, -6.93627655, -7.16326531, -7.3152853 , -7.39233653,
         -7.39441899, -7.32153269, -7.17367763, -6.95085381, -6.65306122,
         -6.28029988, -5.83256976, -5.30987089, -4.71220325, -4.03956685,
         -3.29196168, -2.46938776, -1.57184506, -0.59933361,  0.44814661,
          1.57059559,  2.76801333,  4.04039983,  5.3877551 ,  6.81007913,
          8.30737193,  9.87963349, 11.52686381, 13.24906289, 15.04623074,
         16.91836735, 18.86547272, 20.88754686, 22.98458975, 25.15660142,
         27.40358184, 29.72553103, 32.12244898, 34.59433569, 37.14119117,
         39.76301541, 42.45980841, 45.23157018, 48.07830071, 51.        ]
    )

    np.testing.assert_allclose(mu_szs, precomp_mu_szs)


def test_prob_musz_given_mu_not_negative():
    stacked_model = StackedModel()

    mus = np.random.rand(5)
    mu_szs = np.random.rand(5)
    stacked_model.b_sz = np.random.rand(1)

    a_sz = np.random.rand(1)
    con = 2*np.ones(1)
    radius = None

    stacked_model.set_coords((radius, con, a_sz))

    prob_sz = stacked_model.prob_musz_given_mu(mu_szs, mus)

    assert np.all(prob_sz > 0)


def test_prob_musz_given_mu():
    stacked_model = StackedModel()

    mus = np.linspace(2, 3, num=5)
    mu_szs = np.linspace(2, 3, num=5)
    stacked_model.b_sz = np.linspace(-5, 5, num=5)

    a_sz = np.ones(1)
    con = 2*np.ones(1)
    radius = None

    stacked_model.set_coords((radius, con, a_sz))

    prob_sz = stacked_model.prob_musz_given_mu(mu_szs, mus)

    precomp_prob_sz = np.array(
        [[7.43359757e-06, 1.76297841e-03, 8.76415025e-02, 9.13245427e-01, 1.99471140e+00],
         [6.57000909e-09, 7.43359757e-06, 1.76297841e-03, 8.76415025e-02, 9.13245427e-01],
         [1.21716027e-12, 6.57000909e-09, 7.43359757e-06, 1.76297841e-03, 8.76415025e-02],
         [4.72655194e-17, 1.21716027e-12, 6.57000909e-09, 7.43359757e-06, 1.76297841e-03],
         [3.84729931e-22, 4.72655194e-17, 1.21716027e-12, 6.57000909e-09, 7.43359757e-06]]
    ).T[..., None]

    np.testing.assert_allclose(prob_sz, precomp_prob_sz)


def test_delta_sigma_of_r_divby_nsz():
    """
    This test functions by setting delta_sigma_of_mass to be constant,
    resulting in it being identical to the normalization. Thus this test should
    always return 1s, rather than a true precomputed value
    """

    stacked_model = StackedModel()

    stacked_model.mu_szs = np.linspace(12, 16, 10)
    mus = np.linspace(12, 16, 20)
    stacked_model.mus = mus
    zs = np.linspace(0, 2, 8)
    stacked_model.zs = zs

    stacked_model.dnumber_dlogmass = lambda : np.ones(
        (stacked_model.mus.size, stacked_model.zs.size)
    )

    a_sz = np.ones(1)
    con = 2*np.ones(1)
    rs = np.logspace(-1, 1, 40)

    stacked_model.set_coords((rs, con, a_sz))


    stacked_model.delta_sigma_of_mass = lambda rs,mus,cons,units: np.ones(
        (stacked_model.mus.size, zs.size, rs.size, stacked_model.concentrations.size)
    )

    delta_sigmas = stacked_model.delta_sigma(rs)

    precomp_delta_sigmas = np.ones((rs.size, 1, 1))

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def test_weak_lensing_avg_mass():
    stacked_model = StackedModel()

    stacked_model.mu_szs = np.linspace(12, 16, 10)
    stacked_model.mus = np.linspace(12, 16, 20)
    stacked_model.zs = np.linspace(0, 2, 8)

    a_sz = 2*np.ones(1)
    con = 2*np.ones(1)
    rs = np.logspace(-1, 1, 40)
    stacked_model.set_coords((rs, con, a_sz))

    stacked_model.dnumber_dlogmass = lambda : np.ones(
        (stacked_model.mus.size, stacked_model.zs.size)
    )

    stacked_model.delta_sigma_of_mass = lambda rs,mus,cons: np.ones(
        (stacked_model.mus.size, rs.size)
    )

    avg_wl_mass = stacked_model.weak_lensing_avg_mass()

    assert not np.isnan(avg_wl_mass)


def test_miscentered_delta_sigma():
    stacked_model = StackedModel()

    stacked_model.mu_szs = np.linspace(12, 16, 10)
    zs = stacked_model.zs
    mus = np.array([15])
    stacked_model.mus = mus
    rs = np.logspace(-1, 1, 20)
    cons = np.array([3])
    frac = 0.5

    stacked_model.sigma_of_mass = lambda rs,mus,cons,units: np.ones((mus.size, zs.size, rs.size, cons.size))

    miscentered_sigmas = stacked_model.misc_sigma(rs, mus, cons, frac)

    assert miscentered_sigmas.shape == (1, 20, 20, 1)
