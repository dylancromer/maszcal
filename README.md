# maszcal
Mass SZ Calibrations

## Structure
- Main module code is in `maszcal`.
- `spec` constains function and class specifications (tests of fundamental behaviors, independent of most dependencies. Many of these are unit tests)
-  `tests` contains all other tests including integration tests.

## Dependencies
The module dependencies are:
- dataclasses (if you are using Python version less than 3.7)
- future (Should be removed from this list)
- numpy
- scipy
- astropy
- camb
- [projector](https://github.com/dylancromer/projector)
- [supercubos](https//github.com/dylancromer/supercubos)

In order to run the tests, the minimal requirements are [pytest](https://pytest.org/en/latest/) and [pytest-describe](https://github.com/ropez/pytest-describe). In some cases, I use [pytest-mock](https://pypi.org/project/pytest-mock/) and [pretend](https://github.com/alex/pretend) for stubbing/mocking. Many of the integration tests in `tests` require `matplotlib` for generating plots and use `seaborn` for styling. Running the tests yourself is not necessary; if you do want to run them I suggest focusing on the `spec` tests, which are more valuable for ensuring maszcal is working.
