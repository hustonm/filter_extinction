# Filter extinction

The primary filter for the Roman Galactic Bulge Time Domain Survey, F146, is a very wide filter that extends from ~1-2 microns.
The extinction law changes dramatically over this range, so the extinction in the F146 filter for a given star
is strongly dependent on its spectrum and the absolute extinction. Galactic models typically apply a constant extinction coefficient
using some effective wavelength to
scale A\_Ks or some other absolute extinction quantity to that of each individual filter. Here, we use model spectra and filter integration
to create a more accurate estimate of the extinction coefficient as a function of A\_Ks and stellar colors. For convenience we provide 
corrections the other Roman filters and for both absolute and apparent colors with different observed filter combinations.

This code may be modified to compute corrections for alternate filter sets and/or to use a different extinction law.

`extinction_estimator.py` provides functions to perform the extinction estimation on simulated or observed colors and A\_Ks

`ext_utils.py` contains the functions used to generate the color and extinction grids from model spectra, perform the fits, and to cleanly format the results

`roman_fitting_notebooks` contains Jupyter notebooks where the extinction estimator fits were run and diagnostics were plotted

`roman_fits_abs_AKs5.json` and similar files hold the results of the fits and get used by `extinction_estimator.py`.
