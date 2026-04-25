import json
import numpy as np
import warnings

roman_filter_list = ['f062', 'f087', 'f106', 'f129', 'f158', 'f184', 'f213', 'f146']

# Load up the Roman fit results
with open('roman_fits_abs_AKs5.json', 'r') as f:
    roman_fits_abs_AKs5 = json.load(f)
with open('roman_fits_abs_AKs1.json', 'r') as f:
    roman_fits_abs_AKs1 = json.load(f)
with open('roman_fits_app_AKs5.json', 'r') as f:
    roman_fits_app_AKs5 = json.load(f)
with open('roman_fits_app_AKs1.json', 'r') as f:
    roman_fits_app_AKs1 = json.load(f)

def generic_extinction_polynomial(AKs_C, coeffs, order):
    """
    Generic polynomial function of flexible order. Any number of colors
    may be input, and cross-terms are computed between A_Ks and each color,
    not between different colors.
    
    Identical function to that used to run the fits.

    Parameters:
    -----------
    AKs_C : ndarray
        array of extinction and color values to compute extinction estimate
    coeffs : ndarray
        array of coefficients for the polynomial function
    order : int
        polynomial order for the extinction coefficient function

    Returns:
    --------
    ext_ests : ndarray
        extinction estimates for each star in the AKs_C table
    """
    n_colors = AKs_C.shape[1] - 1 
    n_terms_AKs = order + 1
    n_terms_per_color = order * (order + 1) // 2
    n_terms = n_terms_AKs + (n_colors * n_terms_per_color)
    assert n_terms == len(coeffs)

    AKs = AKs_C[:,0]
    var_terms = [AKs**p for p in range(order+1)]
    n_colors = AKs_C.shape[1]-1
    for i in range(0, n_colors):
        Ci = AKs_C[:, i+1]
        for p in range(1, order+1):
            for q in range(order+1):
                if p + q <= order:
                    var_terms.append((Ci**p) * (AKs**q))
    terms_mat = np.column_stack(var_terms)
    val = terms_mat @ np.array(coeffs)
    return val * AKs


def get_roman_extinction_sim(catalog, low_extinction=False):
    """
    Roman extinction estimator for simulations. Assumes all Roman filter photometry 
    is provided and in absolute AB mags.
    
    Parameters:
    -----------
    catalog : pd.DataFrame, astropy.table.Table, or similar
        required columns: A_Ks, f062, f087, f106, f129, f158, f184, f213, and f146
    low_extinction=False : boolean
        if all A_Ks<=1, use the alternate lower order correction. if any A_Ks>1, a warning
        will be printed, and the higher order correction will be used
        
    Returns:
    --------
    extinctions : dict
        entries of '<filter>':[<ext_star1>, <ext_star2>, ...] for each filter
    """
    # Select the appropriate fit_dict
    fit_dict = roman_fits_abs_AKs5
    if low_extinction and np.all(catalog['A_Ks']<=1):
        fit_dict = roman_fits_abs_AKs1
    elif low_extinction:
        warnings.warn("low_extinction set to True, but some A_Ks > 1. "
                      "switching to 0 <= A_Ks <= 5 fit.")
    
    # Iterate over the filters
    result = {}
    for filt in roman_filter_list:
        filt_fit = fit_dict[filt]
        colors = filt_fit['colors']
        coeffs = filt_fit['coefficients']
        order = filt_fit['order']
        print(f"estimating {filt} extinction using {colors} and order={order} function"

        columns = [catalog['A_Ks']]
        for c in colors:
            f1,f2 = c.split('_')
            columns.append(catalog[f1]-catalog[f2])
        AKs_C = np.stack(columns)
        
        ext_filt = generic_extinction_polynomial(AKs_C, coeffs, order)
        result[filt] = ext_filt
        
    return result
              

              
def get_roman_extinction_obs(catalog, low_extinction=False, nans=None):
    """
    Roman extinction estimator for observations. Assumes all Roman filter photometry 
    is provided and in apparent AB mags.
    
    Parameters:
    -----------
    catalog : pd.DataFrame, astropy.table.Table, or similar
        required columns: A_Ks, f062, f087, f106, f129, f158, f184, f213, and f146, where 
        nan for missing magnitudes are ok (but may result in some nan extinctions)
    low_extinction=False : boolean
        if all A_Ks<=1, use the alternate lower order correction. if any A_Ks>1, a warning 
        will be printed, and the higher order correction will be used
    nans=None : float or int
        fill value that represents nans in the catalog. if none, assumes any NaNs are 
        already represented by np.nan. NaN results will be returned with this fill value too.
        
    Returns:
    --------
    extinctions : dict
        entries of '<filter>':[<ext_star1>, <ext_star2>, ...] for each filter
    """
    # Select the appropriate fit_dict
    fit_dict_all = roman_fits_abs_AKs5
    if low_extinction and np.all(catalog['A_Ks']<=1):
        fit_dict_all = roman_fits_abs_AKs1
    elif low_extinction:
        warnings.warn("low_extinction set to True, but some A_Ks > 1. "
                      "switching to 0 <= A_Ks <= 5 fit.")
    
    # Iterate over the filters for v1 corrections
    filt_dict = filt_dict_all['v1']
    result = {}
    any_nans = False
    for filt in roman_filter_list:
        filt_fit = fit_dict[filt]
        colors = filt_fit['colors']
        coeffs = filt_fit['coefficients']
        order = filt_fit['order']
        print(f"estimating {filt} extinction using {colors} and order={order} function"

        columns = [catalog['A_Ks']]
        for c in colors:
            f1,f2 = c.split('_')
            columns.append(catalog[f1]-catalog[f2])
        AKs_C = np.stack(columns)
        
        ext_filt = generic_extinction_polynomial(AKs_C, coeffs, order)
        result[filt] = ext_filt
        if np.any(np.isnan(ext_filt)):
              any_nans = True
    # Return these if there are no nans, otherwise continue to alternate correcions
    if not any_nans:
        return result
              
    # TODO put in alternate versions for missing photometry
        
    return result