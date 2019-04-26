import numpy as np
from maszcal.model import StackedModel
from maszcal.fitting import Fitter




def test_fitting():
    stacked_model = StackedModel()
    stacked_model.a_sz = np.linspace(-1, 1, 100)

    data = np.loadtxt('data/test/testdata.csv', delimiter=',')

    fitter = Fitter(data = data,
                    likelihood = stacked_model.likelihood,
                    model = stacked_model)

    fit = fitter.do_fit()
