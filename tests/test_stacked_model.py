from maszcal.model import StackedModel
import numpy as np

def test_mu_sz():
    stacked_model = StackedModel()

    mus = np.linspace(1, 10)
    stacked_model.bs = np.linspace(-5,5)
    stacked_model.as_ = np.ones(50)

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


def test_prob_musz_given_mu():
    stacked_model = StackedModel()

    mus = np.linspace(2, 3, num=5)
    mu_szs = np.linspace(2, 3, num=5)
    stacked_model.bs = np.linspace(-5,5, num=5)
    stacked_model.as_ = np.ones(5)

    probs = stacked_model.prob_musz_given_mu(mu_szs, mus)

    precomp_probs = np.array(
        [[7.43359757e-06, 1.76297841e-03, 8.76415025e-02, 9.13245427e-01, 1.99471140e+00],
         [6.57000909e-09, 7.43359757e-06, 1.76297841e-03, 8.76415025e-02, 9.13245427e-01],
         [1.21716027e-12, 6.57000909e-09, 7.43359757e-06, 1.76297841e-03, 8.76415025e-02],
         [4.72655194e-17, 1.21716027e-12, 6.57000909e-09, 7.43359757e-06, 1.76297841e-03],
         [3.84729931e-22, 4.72655194e-17, 1.21716027e-12, 6.57000909e-09, 7.43359757e-06]]
    )

    np.testing.assert_allclose(probs, precomp_probs)


def test_delta_sigma_of_r_dummy():
    stacked_model = StackedModel()

    rs = np.logspace(0, 1, 40)
    stacked_model.delta_sigma_of_mass = lambda rs,mus: np.ones((rs.size, stacked_model.zs.size,  stacked_model.mus.size))

    delta_sigmas = stacked_model.delta_sigma(rs)

    precomp_delta_sigmas = np.array([])

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)
