from spisea import synthetic, atmospheres, reddening
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os.path
import copy
import pandas as pd
import tqdm

# Some simple filter functions for convenience
def get_eff_lam(flt_name):
    """
    Get Vega flux-scaled effective wavelength for a filter
    """
    filt = synthetic.get_filter_info(flt_name)
    obs = pysynphot.Observation(pysynphot.Vega, filt).efflam()
    assert str(filt.waveunits) == 'angstrom'
    return obs

def get_piv_lam(flt_name):
    """
    Get the pivot wavelength for a filter
    """
    filt = synthetic.get_filter_info(flt_name)
    obs = filt.pivot()
    assert str(filt.waveunits) == 'angstrom'
    return obs

def get_avg_lam(flt_name):
    """
    Get the average wavelength for a filter
    """
    filt = synthetic.get_filter_info(flt_name)
    obs = filt.avgwave()
    assert str(filt.waveunits) == 'angstrom'
    return obs

AKs_grid_default = np.round(np.concatenate([np.arange(0.0,0.501, 0.01), np.arange(0.55,2.01, 0.05),
                              np.arange(2.1,5.01, 0.1)]),3)

"""
Parameters:
-----------
A_Ks : list or ndarray
    extinction values in A_Ks to compute grid for
metallicity : float
    [Fe/H] for stellar models
loggs : list of float
    log surface gravity for stellar models
color_filter_list : list of str
    obsstr of filters for color calculation. colors are calculated for each 
    consecutive filter pair in order.
ext_only_filter_list : list of str
    obsstr of filters to exclude from colors but compute extinction for
grid_dir : str or None
    directory to save table in (use None to not save)
recompute_grid : boolean
    force model rerun instead of reload if true
"""
class ExtinctionCoefficientFitter():

    def __init__(self, color_filter_list=['roman,wfi,f062','roman,wfi,f087','roman,wfi,f106','roman,wfi,f129',
                             'roman,wfi,f158','roman,wfi,f184','roman,wfi,f213'],
                       ext_only_filter_list=['roman,wfi,f146'],
                       AKs_grid = AKs_grid_default,
                       metallicity=0.0, loggs=[4.5,2], grid_dir='./grids', recompute_grid=False,
                       filter_synthpop_columns=["R062", "Z087", "Y106", "J129", "W146", "H158", "F184"]):
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = reddening.RedLawSODC(Rv=2.5)
    
        self.color_filter_list = color_filter_list
        self.ext_only_filter_list = ext_only_filter_list
        self.filters_long = color_filter_list + ext_only_filter_list
        self.filter_objs = [synthetic.get_filter_info(f) for f in self.filters_long]
        self.filters_short = [f.split(',')[-1] for f in self.filters_long]
        self.filter_synthpop_columns = self.filters_short
        if filter_synthpop_columns is not None:
            self.filter_synthpop_columns = filter_synthpop_columns
        self.mag_ab_vega = [synthetic.calc_ab_vega_filter_conversion(filt) for filt in self.filters_long]

        self.AKs_grid = AKs_grid
        self.grid_dir = grid_dir
        self.metallicity = metallicity
        self.loggs = loggs
        self.recompute_grid = recompute_grid
        
        print("Load or generate extinction + colors grid")
        self.load_ext_grid()

    def load_ext_grid(self):
        """
        Generate a grid of stellar colors (absolute and observed) and extinctions.

        Returns:
        --------
        grid_phot : DataFrame
            table of stellar parameters with integrated colors and extinctions
        """
        # Check for saved file
        filename = f'{self.grid_dir}/ext_grid_met_{self.metallicity:1.2f}_AKs_{max(self.AKs_grid):1.2f}.h5'
        if (self.grid_dir is not None) and (not self.recompute_grid) and os.path.isfile(filename):
            print("Found saved file "+filename)
            self.ext_grid = pd.read_hdf(filename)
            if list(np.unique(self.ext_grid['A_Ks']))==list(self.AKs_grid):
                return 
            print("Regenerating grid for new A_Ks list")

        # Generate grid if needed
        data = {f'{self.filters_short[i]}_{self.filters_short[i+1]}':[] for i in range(len(self.color_filter_list)-1)}
        ncol = len(data)
        data.update({f'{self.filters_short[i]}_{self.filters_short[i+1]}_abs':[] for i in range(len(self.color_filter_list)-1)})
        data.update({f'ext_{f}':[] for f in self.filters_short})
        data['Teff'] = []
        data['logg'] = []
        data['A_Ks'] = [] 
        for teff in tqdm.tqdm(np.logspace(np.log10(2_500), np.log10(8_000), 50)):
            for logg in self.loggs:
                spec_base = self.atm_func(metallicity=self.metallicity, temperature=teff, gravity=logg)
                mag_base = {}
                for i,f in enumerate(self.filters_short):
                    mag_base[f] = synthetic.mag_in_filter(spec_base, self.filter_objs[i]) + self.mag_ab_vega[i]
                for AKs in self.AKs_grid:
                    spec = copy.deepcopy(spec_base)  # in erg s^-1 cm^-2 A^-1
                    red = self.red_law.reddening(AKs).resample(spec.wave)
                    spec *= red
                    mag = {}
                    for i,f in enumerate(self.filters_short):
                        mag[f] = synthetic.mag_in_filter(spec, self.filter_objs[i]) + self.mag_ab_vega[i]
                        data[f'ext_{f}'].append(mag[f]-mag_base[f])
                    for i in range(len(self.color_filter_list)-1):
                        f1,f2 = self.filters_short[i], self.filters_short[i+1]
                        c = f'{self.filters_short[i]}_{self.filters_short[i+1]}'
                        data[c].append(mag[f1]-mag[f2])
                        data[c+'_abs'].append(mag_base[f1]-mag_base[f2])
                    data['Teff'].append(int(np.round(teff)))
                    data['logg'].append(logg)
                    data['A_Ks'].append(np.round(AKs,3))

        self.ext_grid = pd.DataFrame(data)
        if self.grid_dir is not None:
            self.ext_grid.to_hdf(filename, key='data', index=False)
        return 
    
    def plot_true_extinction(self, filt, A_Ks):
        if isinstance(A_Ks, float) or isinstance(A_Ks, int):
            A_Ks = [A_Ks]
        fig, ax = plt.subplots(nrows=1, ncols=len(A_Ks),sharey=False, figsize=(5*len(A_Ks),5), layout='constrained')
        for i,AKs in enumerate(A_Ks):
            for logg in self.loggs:
                idxs = (self.ext_grid['A_Ks']==AKs) & (self.ext_grid['logg']==logg)
                ax[i].scatter(self.ext_grid['Teff'][idxs], self.ext_grid[f'ext_{filt}'][idxs], s=5, label=f'logg={logg:.1f}')
            ax[i].set_title(f'A_Ks = {AKs:.2f}')
            ax[i].set_xticks([3e3,1e4,2e4])
            ax[i].legend()
            ax[i].set_xlabel('Teff (K)')
            ax[i].set_xscale('log')
        ax[0].set_ylabel(r'$\Delta$F146$_{true}$')
        os.makedirs('figures',exist_ok=True)
        fig.savefig(f'figures/ext_true_f{filt}.png')
        return fig,ax

    
    @staticmethod
    def generic_extinction_polynomial(AKs_C, coeffs, order):
        """
        Generic polynomial function of flexible order. Any number of colors
        may be input, and cross-terms are computed between A_Ks and each color,
        not between different colors.

        Parameters:
        -----------
        AKs_C : ndarray
            array of extinction and color values to compute extinction estimate
        coeffs : ndarray
            array of coefficients for the polynomial function
        order : int
            polynomial order for the extinction coefficient function

        Returns:
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
    
    def run_fit(self, filt, colors, order=4):
        """
        Fit runner for the generic extinction polynomial function

        Parameters:
        -----------
        filt : str
            filter short string to fit extinction coefficient function
        colors : list of str
            color short strings to use as input parameters in the function
        grid : DataFrame
            table output from get_large_ext_grid
        order : str
            polynomial order for extinction coefficient function

        Returns:
        --------
        best_fit_function : function
            function to compute best-fit extinction estimate
        AKs_C : ndarray
            numpy array of A_Ks values and colors used in the fit
        ext_ests : ndarray
            extinction estimates for the input table stars using the best-fit function
        fit_ext_arr : ndarray
            true extinction values used in the fit
        coeffs : ndarray
            best fit polynomial coefficients
        """
        self.ext_fit_filter = filt
        self.ext_fit_colors = colors
        self.filt_ext_arr = self.ext_grid[f'ext_{filt}'].to_numpy()
        self.AKs_C = self.ext_grid[['A_Ks']+colors].to_numpy()
        self.n_colors = self.AKs_C.shape[1] - 1 
        self.order = order
        self.n_terms_AKs = self.order + 1
        self.n_terms_per_color = self.order * (self.order + 1) // 2
        self.n_terms = self.n_terms_AKs + (self.n_colors * self.n_terms_per_color)

        def fit_wrapper(AKs_C_arr, *coeffs):
            return self.generic_extinction_polynomial(AKs_C_arr, coeffs, order)

        res = curve_fit(fit_wrapper, self.AKs_C, self.filt_ext_arr, p0=np.ones(self.n_terms))
        ext_ests = self.generic_extinction_polynomial(self.AKs_C, res[0], order=order)

        def result_wrapper(AKs_C_arr):
            return self.generic_extinction_polynomial(AKs_C_arr, res[0], order)
        
        self.best_fit_function = result_wrapper
        self.ext_ests = result_wrapper(self.AKs_C)
        self.best_fit_coeffs = res[0]
        self.best_fit_coeffs_cov = res[1]
        return 
    
    def plot_fit_result(self):
        if not hasattr(self, 'best_fit_function'):
            raise RuntimeError("A fit must be run before the results can be plotted.")
        plt.rcParams['font.size'] = 14
        fig,ax = plt.subplots(nrows=1, ncols=len(self.loggs), sharey=True, 
                              figsize=(5*len(self.loggs),5), layout='constrained')

        for i in range(len(self.loggs)):
            idxs = self.ext_grid['logg']==self.loggs[i]
            im1 = ax[i].scatter(self.ext_grid['Teff'][idxs], self.ext_ests[idxs]-self.filt_ext_arr[idxs], 
                    c=self.ext_grid['A_Ks'][idxs], s=5)
            ax[i].set_title(f'logg={self.loggs[i]}')
            ax[i].set_xlabel('Teff (K)')
            ax[i].set_xscale('log')

        fig.colorbar(im1, ax=ax[1], label='AKs')
        ax[0].set_ylabel(r'$\Delta$'+self.ext_fit_filter+r'$_{\rm fit}$'+
                         r'- $\Delta$'+self.ext_fit_filter+r'$_{\rm true}$')
        fig.savefig(f'figures/ext_corr_grid_{self.ext_fit_filter}.png')
        return fig,ax

    def get_catalog_true(self, catalog):
        cat = catalog.copy()
        for i in range(len(self.color_filter_list)-1):
            col = f'{self.filters_short[i]}_{self.filters_short[i+1]}_abs_syn'
            f1, f2 = self.filter_synthpop_columns[i], self.filter_synthpop_columns[i+1]
            if (f1 in cat.columns) and (f2 in cat.columns):
                cat.loc[:,col] = cat[f1]-cat[f2]

        cat.loc[:,'Teff'] = 10**cat['log_Teff']
        for filt in self.filters_short:
            cat.loc[:,f'ext_{filt}_true'] = np.nan
        for i in tqdm.tqdm(cat.index):
            spec_base = self.atm_func(metallicity=cat['Fe/H_initial'][i],
                                 temperature=cat['Teff'][i], 
                                 gravity=cat['log_g'][i], verbose=False)
            spec = copy.deepcopy(spec_base)  # in erg s^-1 cm^-2 A^-1
            red = self.red_law.reddening(cat['A_Ks'][i]).resample(spec.wave)
            spec *= red
            mag_base = {}
            mag = {}
            for i,f in enumerate(self.filters_short):
                mag_base[f] = synthetic.mag_in_filter(spec_base, self.filter_objs[i]) + self.mag_ab_vega[i]
                mag[f] = synthetic.mag_in_filter(spec, self.filter_objs[i]) + self.mag_ab_vega[i]
                cat.loc[i, f'ext_{f}_true'] = mag[f]-mag_base[f]
            for i in range(len(self.color_filter_list)-1):
                f1,f2 = self.filters_short[i], self.filters_short[i+1]
                c = f'{self.filters_short[i]}_{self.filters_short[i+1]}'
                cat.loc[i,c+'_app_spi'] = (mag[f1]-mag[f2])
                cat.loc[i,c+'_abs_spi'] = (mag_base[f1]-mag_base[f2])
        return cat