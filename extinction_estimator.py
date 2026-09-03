import json
import numpy as np
import warnings
import os

def convert_magsys(input_catalog, input_magsys)
    with open("magsys_conversions.json", "r") as f:
        magsys_conversions = json.load(f)
    catalog = {}
    for col in input_catalog:
        if col in magsys_conversions['AB']:
            conv = magsys_conversions['AB'][col]
            if input_magsys=='ST':
                conv -= magsys_conversions['ST'][col]
            catalog[col] = np.array(input_catalog[col]) + conv
    return catalog 

def get_table(file='extinction_correction_table.dat'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ext_cor_tab = Table.read(f'{current_dir}/{file}', format='ascii.mrt')
    ext_cor_tab = ext_cor_tab.filled(999)
    fit_dict = {}
    color_cols = [col for col in ext_cor_tab.colnames if col.startswith('Color')]
    a_cols = [col for col in ext_cor_tab.colnames if col.startswith('a_')]
    coeff_cols = [col for col in ext_cor_tab.colnames if (col[:2] in ['a_', 'b_'])]
    fit_order = len(a_cols)-1
    for i in range(len(ext_cor_tab)):
        filt = ext_cor_tab['Filter'][i]
        fit_dict[filt] = {}
        fit_dict[filt]['colors'] = [ext_cor_tab[col][i] for col in color_cols if (ext_cor_tab[col][i]!="999")]
        fit_dict[filt]['order'] = self.fit_order
        fit_dict[filt]['coefficients'] = [ext_cor_tab[col][i] for col in coeff_cols if ~(ext_cor_tab[col][i]==999.0)]
    return fit_dict

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

def calc_corr(catalog, filt_fit):
    colors = filt_fit['colors']
    coeffs = filt_fit['coefficients']
    order = filt_fit['order']
    print(f"Estimating {filt} extinction using {colors} and order={order} function")

    columns = [np.array(catalog['A_Ks'])]
    for c in colors:
        f1,f2,_ = c.split('-')
        columns.append(np.array(catalog[f1])-np.array(catalog[f2]))
    AKs_C = np.stack(columns,axis=1)
    
    ext_filt = generic_extinction_polynomial(AKs_C, coeffs, order)
    return ext_filt


def estimate_extinction(input_catalog, appmag=None, magsys=None, 
                        low_extinction=False):
    """
    Roman extinction estimator for observations.
    
    Parameters:
    -----------
    input_catalog : pd.DataFrame, astropy.table.Table, or similar
        Requires A_Ks column + magnitudes for relevant filter sets. Magnitudes can be 
        nan but may produce nan extinction estimates.
    appmag : boolean or None
        True for apparent mags, False for absolute. Assumes absolute and raises warning 
        if None.
    magsys : str or None
        Magnitude system for the catalog ('AB', 'Vega', or 'ST'). Assumes AB and raises a
        warning if None.
    low_extinction=False : boolean
        If all A_Ks<=1, use the alternate lower order correction. if any A_Ks>1, a warning 
        will be printed, and the higher order correction will be used
        
    Returns:
    --------
    extinctions : dict
        entries of '<filter>':[<ext_star1>, <ext_star2>, ...] for each filter
    """

    # Magnitude system handling
    if magsys is None:
        warnings.warn("Input magsys not specified. Assuming AB.")
        magsys = 'AB'
    if magsys == 'AB':
        catalog = input_catalog
    elif magsys in ['Vega', 'ST']
        catalog = convert_magsys(input_catalog, magsys)
    else:
        raise ValueError(f"Invalid magsys {magsys}. Use 'AB', 'Vega', or 'ST'.")

    if appmag is None:
        warnings.warn("Absolute versus apparent not specified. Assuming AB.")
        appmag = False
    
    # Check filters
    filter_list = [f for f in catalog if (f in fit_dict.keys())]
    invalid_filter_list = [f for f in catalog if (f not in fit_dict.keys())]
    if len(filter_list)==0:
        raise ValueError(f"No valid filter columns found. Valid filters are: {fit_dict.keys()}")
    if len(invalid_filter_list)>0:
        warnings.warn(f"No extinction estimate available for columns {invalid_filter_list}")

    # Get the extinction estimates by filter
    result = {}
    for filt in filter_list:
        result['A_'+filt] = calc_corr(catalog, fit_dict[filt])

    # TODO multi-iteration apparent mag estimation
        
    return result
              


# TODO consolidate into generic function
def filter_extinction_obs(input_catalog, magsys=None, low_extinction=False, nans=None):
    """
    Roman extinction estimator for observations.
    
    Parameters:
    -----------
    catalog : pd.DataFrame, astropy.table.Table, or similar
        required columns: A_Ks, f062, f087, f106, f129, f158, f184, f213, and f146, where 
        nan for missing magnitudes are ok (but may result in some nan extinctions)
    magsys : str or None
        magnitude system for the catalog ('AB', 'Vega', or 'ST'). assumes AB and raises a
        warning if None.
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

    # Magnitude system handling
    if magsys is None:
        warnings.warn("Input magsys not specified. Assuming AB.")
        magsys = 'AB'
    if magsys == 'AB':
        catalog = input_catalog
    elif magsys in ['Vega', 'ST']
        catalog = convert_magsys(input_catalog, magsys)
    else:
        raise ValueError(f"Invalid magsys {magsys}. Use 'AB', 'Vega', or 'ST'.")

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
        print(f"estimating {filt} extinction using {colors} and order={order} function")

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
