# maszcal
maszcal is a package dedicated to calibration of the observable-mass relation for galaxy clusters, with a focus on the thermal Sunyaev-Zeldovich signal's relation to mass. maszcal is designed to explicitly model baryonic matter density profiles, differing from most previous approaches that treat galaxy clusters as purely dark matter. To do this, it uses a generalized Nararro-Frenk-White (GNFW) density to represent the baryons, while using the more typical NFW profile to represent dark matter. The theoretic methodology is detailed in the accompanying paper (ArXiv link pending).

maszcal also uses a Principle Component Analysis (PCA) emulator (AKA a surrogate model) based on the Coyote Universe emulator. The emulator and PCA code is contained in dependencies, but it is crucial to running the MCMC analysis much faster than would be possible with the full model.

## Structure
- Main module code is in `maszcal`
- `spec` contains fast-running tests (mostly unit tests) which are designed to ensure operability of the code
- `int` contains slow-running tests that check numerical results or produce plots for visual examination

## Installation
For now, I recommend installing using `pip`: run
```
pip install --user -e .
```
inside the base directory. To install dependencies, run
```
pip install -r requirements.txt
```

## Dependencies
The module dependencies are:
- dataclasses (if you are using Python version less than 3.7)
- colossus
- numpy
- scipy
- astropy
- camb
- scikit-learn
- [mcfit](https://github.com/dylancromer/mcfit) (forked from [eelregit/mcfit](https://github.com/eelregit/meso)) -> calculates correlation function from power spectrum
- [pality](https://github.com/dylancromer/pality) -> provides PCA calculation
- [ostrich](https://github.com/dylancromer/ostrich) -> provides a PCA emulator
- [meso](https://github.com/dylancromer/meso) -> provides miscentering
- [projector](https://github.com/dylancromer/projector) -> provides projection from density to excess surface density
- [supercubos](https://github.com/dylancromer/supercubos) -> calculates Latin Hypercubes
- [smolyak](https://github.com/dylancromer/smolyak) -> provides Smolyak grid interpolation

In order to run the tests, the minimal requirements are [pytest](https://pytest.org/en/latest/) and [pytest-describe](https://github.com/ropez/pytest-describe). In some cases, I use [pytest-mock](https://pypi.org/project/pytest-mock/) and [pretend](https://github.com/alex/pretend) for stubbing/mocking. Many of the integration tests in `tests` require Matplotlib for generating plots and use Seaborn for styling. Running the tests yourself is not necessary, but can be done via
```
pytest spec/**/*.py
```
if you wish.

If you want to use my analysis scripts, you will also need to install [pathos](https://github.com/uqfoundation/pathos) and [emcee](https://emcee.readthedocs.io/en/stable/). These can both be installed via `pip install --user pathos emcee`.

## Usage
maszcal contains all of the classes and functions needed to run a full mass calibration analysis that includes baryons. Right now, you can see examples of this in the `scripts` directory. In the next couple of weeks, a tutorial Jupyter notebook will be provided with some example data to fit.

## Future development
maszcal is currently in a beta release state (version 0.9). Over the next 2-3 months, a large number of changes are going to be made. The planned changed are as follows.

### A Public API
Right now if you want to use maszcal, you need to glue a bunch of classes together yourself, and do the MCMC fitting with a third-party library. The public API will funnel the whole analysis process through a single public class that will be considerably easier to use if you aren't familiar with the maszcal internals.

### Documentation
Right now, the module lacks much inline documentation. Once the public API is finalized, auto-generated documentation will be provided to help both users and those who wish to help develop the package.

### Cython Extension
One of the biggest changes is that in the near-future, maszcal will contain part of its core library as a Cython extension. This is done because including scatter and miscentering in the observable-mass relation adds a considerable memory overhead to calculating the model, one which is basically insurmountable on a personal computer. 

This is because both of these corrections require numerically integrating the model. Doing these integrations by a loop is slow (far, far too slow); doing them via  numpy requires a lot of memory (far, far too much). This problem seems to be outside the realm of what can be solved within just the Python + Numpy ecosystem, and requires creating either a C or Cython extension module. I have opted for the latter.

maszcal will continue to be easy to install even after this extension is added, but if you intend to develop with maszcal, you should be aware of this planned change.

## Notes

### Little Hubble Conventions
maszcal doesn't use so-called "little-h" (the Hubble constant divided by 100) factors at all. Masses are masses, radii and radii, etc. In order to get consistent results, you must remove all h-factors from anything you input, and if you want them on the output values you will have to re-add them yourself too. [But then again, little-h probably shouldn't be used anyway.](https://arxiv.org/pdf/1308.4150.pdf)
