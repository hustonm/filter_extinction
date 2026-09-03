from spisea import synthetic, atmospheres, reddening
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial
import os.path
import copy
import pandas as pd
import tqdm
import string
import pdb
import re
import astropy.units as u
from astropy.table import Table, Column, MaskedColumn
import warnings, contextlib
import json
plt.rcParams['font.size'] = 16

# Some simple filter functions for convenience
def get_eff_lam(flt_name):
    import synphot
    from spisea.synthetic import vega
    """
    Get Vega flux-scaled effective wavelength for a filter
    """
    filt = synthetic.get_filter_info(flt_name)
    obs = synphot.Observation(vega, filt
                    ).effective_wavelength()
    return obs.to(u.micron).value

def get_piv_lam(flt_name):
    """
    Get the pivot wavelength for a filter
    """
    filt = synthetic.get_filter_info(flt_name)
    obs = filt.pivot()
    return obs.to(u.micron).value

def get_avg_lam(flt_name):
    """
    Get the average wavelength for a filter
    """
    filt = synthetic.get_filter_info(flt_name)
    obs = filt.avgwave()
    return obs.to(u.micron).value

AKs_grid_default = np.round(np.arange(0.1,5.01, 0.1),3)
red_law_default = reddening.get_red_law('SODC,2.5')

class ExtinctionCoefficientFitter():

    def __init__(self, color_filter_list=[['roman,wfi,f062','roman,wfi,f087','roman,wfi,f106','roman,wfi,f129',
                             'roman,wfi,f158','roman,wfi,f184','roman,wfi,f213','roman,wfi,f146']],
                       ext_only_filter_list=['euclid,vis'],
                       AKs_grid = AKs_grid_default,
                       metallicity=0.0, loggs=[4.5,2], grid_dir='./grids', recompute_grid=False,
                       filter_synthpop_columns=["R062", "Z087", "Y106", "J129", "F184", "H158", None, "W146", None],
                       red_law=red_law_default,
                       figure_dir='figures'):
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
        red_law : SPISEA reddening law object
            reddening law to apply to spectra
        """
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = red_law
        self.figure_dir = figure_dir+'/'
    
        self.color_filter_list = color_filter_list
        self.ext_only_filter_list = ext_only_filter_list
        self.filters_long = color_filter_list[0].copy()
        for cfl in color_filter_list[1:]:
            self.filters_long += cfl
        self.filters_long += self.ext_only_filter_list
        self.filter_objs = [synthetic.get_filter_info(f) for f in self.filters_long]
        self.filters_short = ['_'.join(f.split(',')) for f in self.filters_long]
        self.filter_synthpop_columns = self.filters_short
        if filter_synthpop_columns is not None:
            self.filter_synthpop_columns = filter_synthpop_columns
        self.mag_ab_vega = [synthetic.calc_ab_vega_filter_conversion(filt).value for filt in self.filters_long]

        self.AKs_grid = AKs_grid
        self.grid_dir = grid_dir
        self.metallicity = metallicity
        self.loggs = loggs
        self.recompute_grid = recompute_grid
        
        print("Load or generate extinction + colors grid")
        self.load_ext_grid()
        self.ext_grid.loc[:,'neg_A_Ks'] = -self.ext_grid['A_Ks']
        self.ext_grid.sort_values(by=['neg_A_Ks', 'logg', 'Teff'], inplace=True)
        self.ext_grid.drop(columns=['neg_A_Ks'], inplace=True)
        self.ext_grid.reset_index(inplace=True, drop=True)

    def make_mag_conversion_json(self, json_file='../magsys_conversions.json'):
        self.mag_st_vega = [synthetic.calc_st_vega_filter_conversion(filt).value for filt in self.filters_long]
        conv_dict = {"AB":dict(zip(self.filters_short, self.mag_ab_vega)),
                     "ST":dict(zip(self.filters_short, self.mag_st_vega))}
        with open(json_file, 'w') as f:
            json.dump(conv_dict, f, indent=4)

        
    def load_ext_grid(self):
        """
        Generate a grid of stellar colors (absolute and observed) and extinctions.

        Returns:
        --------
        grid_phot : DataFrame
            table of stellar parameters with integrated colors and extinctions
        """
        # Check for saved file
        filename = f'{self.grid_dir}/ext_grid_met_{self.metallicity:1.2f}_AKs_{max(self.AKs_grid):1.2f}_'+\
                    f'{self.red_law.__class__.__name__}.h5'
        if (self.grid_dir is not None) and (not self.recompute_grid) and os.path.isfile(filename):
            print("Found saved file "+filename)
            self.ext_grid = pd.read_hdf(filename)
            self.ext_grid = self.ext_grid[self.ext_grid['A_Ks']>0.0].copy()
            if list(np.sort(np.unique(self.ext_grid['A_Ks'])))==list(np.sort(self.AKs_grid)):
                return 
            print("Regenerating grid for new A_Ks list")

        # Generate grid if needed
        data = {}
        i=0
        for fset in self.color_filter_list:
            for j in range(len(fset)-1):
                data[f'{self.filters_short[i]}-{self.filters_short[i+1]}'] = [] 
                data[f'{self.filters_short[i]}-{self.filters_short[i+1]}-abs'] = []
                i += 1
            i+=1
        data.update({f'ext_{f}':[] for f in self.filters_short})
        data['Teff'] = []
        data['logg'] = []
        data['A_Ks'] = [] 
        for teff in tqdm.tqdm(np.logspace(np.log10(2_500), np.log10(10_000), 100)):
            for logg in self.loggs:
                spec_base = self.atm_func(metallicity=self.metallicity, temperature=teff, gravity=logg)
                mag_base = {}
                for i,f in enumerate(self.filters_short):
                    mag_base[f] = synthetic.mag_in_filter(spec_base, self.filter_objs[i]) + self.mag_ab_vega[i]
                for AKs in self.AKs_grid:
                    spec = copy.deepcopy(spec_base)  # in erg s^-1 cm^-2 A^-1
                    red = self.red_law.extinction_curve(AKs,spec.waveset)
                    spec *= red
                    mag = {}
                    for i,f in enumerate(self.filters_short):
                        mag[f] = synthetic.mag_in_filter(spec, self.filter_objs[i]) + self.mag_ab_vega[i]
                        data[f'ext_{f}'].append(mag[f]-mag_base[f])
                    i=0
                    for fset in self.color_filter_list:
                        for j in range(len(fset)-1):
                            f1,f2 = self.filters_short[i], self.filters_short[i+1]
                            c = f'{self.filters_short[i]}-{self.filters_short[i+1]}'
                            data[c].append(mag[f1]-mag[f2])
                            data[c+'-abs'].append(mag_base[f1]-mag_base[f2])
                            i += 1
                        i += 1
                    data['Teff'].append(int(np.round(teff)))
                    data['logg'].append(logg)
                    data['A_Ks'].append(np.round(AKs,3))

        self.ext_grid = pd.DataFrame(data)
        if self.grid_dir is not None:
            self.ext_grid.to_hdf(filename, key='data', index=False)
        return 
    
    def derive_color(self, col_eqn):
        """
        Derive a color with a filter combination not produced by default.
        
        Parameters:
        -----------
        col_eqn : str
            string equation evaluate the color from existing columns
        """
        self.ext_grid.eval(col_eqn, inplace=True)
        return

    
    def plot_true_extinction(self, filt, A_Ks):
        if isinstance(A_Ks, float) or isinstance(A_Ks, int):
            A_Ks = [A_Ks]
        fig, ax = plt.subplots(nrows=1, ncols=len(A_Ks),sharey=False, figsize=(5*len(A_Ks),5), layout='constrained')
        for i,AKs in enumerate(A_Ks):
            for logg in self.loggs:
                idxs = (self.ext_grid['A_Ks']==AKs) & (self.ext_grid['logg']==logg)
                ax[i].scatter(self.ext_grid['Teff'][idxs], self.ext_grid[f'ext_{filt}'][idxs], s=5, label=f'logg={logg:.1f}')
            ax[i].set_title(fr'A$_{{\rm Ks}}$ = {AKs:.2f}', loc='left')
            ax[i].set_xticks([3e3,1e4,2e4])
            ax[i].set_xlabel(r'T$_{\rm eff}$ (K)')
            ax[i].set_xscale('log')
            lam = get_eff_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks = getattr(self.red_law, self.red_law.name.split(',')[0])(lam, AKs)[0]
            textstr = r'$A_{\lambda,eff}$ = ' + f'{alam_aks:.3f}'
            lam = get_piv_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks = getattr(self.red_law, self.red_law.name.split(',')[0])(lam, AKs)[0]
            textstr += '\n'+r'$A_{\lambda,piv}$ = ' + f'{alam_aks:.3f}'
            lam = get_avg_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks = getattr(self.red_law, self.red_law.name.split(',')[0])(lam, AKs)[0]
            textstr += '\n'+r'$A_{\lambda,avg}$ = ' + f'{alam_aks:.3f}'
            ax[i].text(0.05, 0.95, textstr, transform=ax[i].transAxes,
                    verticalalignment='top')
        ax[0].legend(loc=4)
        ax[0].set_ylabel(fr'$A_{{{filt.split("_")[-1]},true}}$')
        
        os.makedirs('figures',exist_ok=True)
        print("saving figure")
        fig.savefig(f'{self.figure_dir}/ext_true_{filt}.png')
        return fig,ax
    
    def plot_extinction_difference(self, filt, A_Ks, show_msos_version=False):
        if isinstance(A_Ks, float) or isinstance(A_Ks, int):
            A_Ks = [A_Ks]
        fig, ax = plt.subplots(nrows=1, ncols=len(A_Ks),sharey=False, figsize=(5*len(A_Ks),5), layout='constrained')
        for i,AKs in enumerate(A_Ks):
            ax[i].set_title(fr'A$_{{\rm Ks}}$ = {AKs:.2f}', loc='left')
            ax[i].set_xticks([2e3,5e3,1e4])
            ax[i].set_xlabel(r'T$_{\rm eff}$ (K)')
            ax[i].set_xscale('log')
            lam_eff = get_eff_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks_eff = getattr(self.red_law, self.red_law.name.split(',')[0])(lam_eff, AKs)[0].value
            lam_piv = get_piv_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks_piv = getattr(self.red_law, self.red_law.name.split(',')[0])(lam_piv, AKs)[0].value
            lam_avg = get_avg_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks_avg = getattr(self.red_law, self.red_law.name.split(',')[0])(lam_avg, AKs)[0].value
            if show_msos_version:
                filtobj = synthetic.get_filter_info(self.filters_long[self.filters_short.index(filt)])
                red_law_filt = getattr(self.red_law, self.red_law.name.split(',')[0])(filtobj.wave, AKs)
                alam_aks_est = np.sum(filtobj.throughput * red_law_filt)/np.sum(filtobj.throughput)
                print(np.sum(filtobj.throughput * red_law_filt)/np.sum(filtobj.throughput),alam_aks_est)
            
            ax[i].plot([2500,10_000], [0,0], c='gray')
            for j,logg in enumerate(self.loggs):
                idxs = (self.ext_grid['A_Ks']==AKs) & (self.ext_grid['logg']==logg)
                ls = ['-',':'][j]
                lab = [fr'$\lambda_{{\rm eff,Vega}}$ ({lam_eff:.3f} $\mu$m)', None][j]
                ax[i].plot(self.ext_grid['Teff'][idxs], alam_aks_eff -  self.ext_grid[f'ext_{filt}'][idxs], 
                           label=lab,
                           linestyle=ls, c='C0')
                lab = [fr'$\lambda_{{\rm pivot}}$ ({lam_piv:.3f} $\mu$m)', None][j]
                ax[i].plot(self.ext_grid['Teff'][idxs], alam_aks_piv -  self.ext_grid[f'ext_{filt}'][idxs], 
                           label=lab,
                           linestyle=ls, c='C1')
                lab = [fr'$\lambda_{{\rm average}}$ ({lam_avg:.3f} $\mu$m)', None][j]
                ax[i].plot(self.ext_grid['Teff'][idxs], alam_aks_avg -  self.ext_grid[f'ext_{filt}'][idxs], 
                           label=lab,
                           linestyle=ls, c='C2')
                if show_msos_version:
                    lab = [fr'MSOS estimate method', None][j]
                    ax[i].plot(self.ext_grid['Teff'][idxs], alam_aks_est -  self.ext_grid[f'ext_{filt}'][idxs], 
                               label=lab,
                               linestyle=ls, c='C3')

        ax[1].legend(fontsize=14)#loc=1)
        ax[0].set_ylabel(fr'$A_{{\lambda, eff}} - A_{{{filt.split("_")[-1]},true}}$')
        ax[0].set_xlim(2500,10_000)
        ax[1].set_xlim(2500,10_000)
        
        os.makedirs('figures',exist_ok=True)
        fig.savefig(f'{self.figure_dir}/ext_true_diff_{filt}.png')
        return fig,ax
    
    def evaluate_extinction_difference(self, filt, A_Ks):
        lam_eff = get_eff_lam(self.filters_long[self.filters_short.index(filt)])
        lam_piv = get_piv_lam(self.filters_long[self.filters_short.index(filt)])
        lam_avg = get_avg_lam(self.filters_long[self.filters_short.index(filt)])
        alam_aks_eff = getattr(self.red_law, self.red_law.name.split(',')[0])(lam_eff, 1)[0].value
        alam_aks_piv = getattr(self.red_law, self.red_law.name.split(',')[0])(lam_piv, 1)[0].value
        alam_aks_avg = getattr(self.red_law, self.red_law.name.split(',')[0])(lam_avg, 1)[0].value
            
        AKs = self.ext_grid['A_Ks']
        mxs = []
        mxs.append(np.max(np.abs(alam_aks_eff -  self.ext_grid[f'ext_{filt}']/AKs)/(self.ext_grid[f'ext_{filt}']/AKs)))
        mxs.append(np.max(np.abs(alam_aks_piv -  self.ext_grid[f'ext_{filt}']/AKs)/(self.ext_grid[f'ext_{filt}']/AKs)))
        mxs.append(np.max(np.abs(alam_aks_avg -  self.ext_grid[f'ext_{filt}']/AKs)/(self.ext_grid[f'ext_{filt}']/AKs)))
                
        sstr = 'NO CORR NEEDED' if (np.max(mxs)<0.05) else ''
        print(filt, np.max(mxs), sstr)
        
        return 

    
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
        return val
    
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

        res = curve_fit(fit_wrapper, self.AKs_C, self.filt_ext_arr/self.AKs_C[:,0], p0=np.ones(self.n_terms))
        ext_ests = self.generic_extinction_polynomial(self.AKs_C, res[0], order=order)

        def result_wrapper(AKs_C_arr):
            return self.generic_extinction_polynomial(AKs_C_arr, res[0], order)
        
        self.best_fit_function = result_wrapper
        self.ext_ests = result_wrapper(self.AKs_C) * self.AKs_C[:,0]
        self.best_fit_coeffs = res[0]
        self.best_fit_coeffs_cov = res[1]
        return 
    
    def get_fit_dict(self):
        return {self.ext_fit_filter: {"colors": self.ext_fit_colors,
                                      "coefficients": list(self.best_fit_coeffs),
                                      "order": self.order}}
    
    def print_fit_json(self):
        color_str = '['
        for i,color in enumerate(self.ext_fit_colors):
            one_color_str = '_'.join(color.split('_')[:2])
            color_str += f'\"{one_color_str}\"'
            if i+1<self.n_colors:
                color_str += ', '
        color_str += ']'
        
        coeff_str = '['
        for i,coeff in enumerate(self.best_fit_coeffs):
            coeff_str += f'{coeff:.9e}'
            if i+1<self.n_terms:
                coeff_str += ', '
            if (i>0) and ((i+1)%5==0) and (i+1<self.n_terms):
                coeff_str += '\n\t\t\t\t'
        coeff_str += ']\n'
        
        
        json_str = f'\t\"{self.ext_fit_filter}\": {{\"colors\": {color_str},\n' \
                            f'\t\t\"coefficients\": {coeff_str}' \
                            f'\t}},'
        print(json_str)
    
    def print_function_latex(self):
        """
        Print the functional form of the polynomial equation in latex format.
        
        An llm was used to generate the first draft of this function.
        """
        order=self.order
        colors=self.ext_fit_colors
        filt = self.ext_fit_filter
        aks = "A_{Ks}"
        result_var = f"\\frac{{A_{{{filt}}}}}{{{aks}}}"

        # Use lowercase alphabet for coefficient letters: a, b, c, d...
        letters = string.ascii_lowercase 

        all_groups = []

        # 1. Base AKs terms (Group -1) - Uses 'a'
        aks_only = []
        for p in range(order + 1):
            coeff = f"a_{{{p}}}"
            term = coeff + (f" {aks}^{{{p}}}" if p > 1 else (f" {aks}" if p == 1 else ""))
            aks_only.append(term)
        all_groups.append(aks_only)

        # 2. Color cross terms (Group 0, 1, 2...) - Uses 'b', 'c', 'd'...
        for i in range(len(colors)):
            ci = f"C_{{{i+1}}}"
            letter = letters[i + 1] # Skip 'a'
            color_terms = []
            # We need a counter for the subscript within this letter group
            sub_idx = 0
            for p in range(1, order + 1):
                for q in range(order + 1):
                    if p + q <= order:
                        term_str = f"{letter}_{{{sub_idx}}}"
                        term_str += f" {ci}^{{{p}}}" if p > 1 else f" {ci}"
                        if q > 0:
                            term_str += f" {aks}^{{{q}}}" if q > 1 else f" {aks}"
                        color_terms.append(term_str)
                        sub_idx += 1
            all_groups.append(color_terms)

        # 3. Format into an aligned block
        lines = [" + ".join(group) for group in all_groups]

        formatted_lines = [f"{result_var} = {lines[0]}"]
        for line in lines[1:]:
            formatted_lines.append(f"\\quad + {line}")

        equation_body = " \\\\ \n".join(formatted_lines)
        return f"\\begin{{aligned}} \n {equation_body} \n \\end{{aligned}}"
    
    




    def print_coeffs_deluxetable(self, precision=5, include_errors=True):
        """
        Generates a LaTeX deluxetable for the current fit results and returns it as a string.
        
        An llm was used to generate the first draft of this function.

        Parameters:
        -----------
        precision : int, default 5
            Number of decimal places for values and errors.
        include_errors : bool, default True
            Whether to include the 'Uncertainty' column derived from the 
            covariance matrix.

        Returns:
        --------
        table_str : str
            A complete LaTeX deluxetable string including caption, headers, 
            formatted data rows, and table comments.
        """
        letters = string.ascii_lowercase
        aks = "A_{Ks}"
        coeffs = self.best_fit_coeffs
        output = []
        
        # 1. Prepare Column Setup
        col_def = "{lccc}" if include_errors else "{lcc}"
        header = "\\tablehead{\\colhead{ID} & \\colhead{Term} & \\colhead{Value}"
        if include_errors:
            header += " & \\colhead{Uncertainty}}"
            errors = np.sqrt(np.diag(self.best_fit_coeffs_cov))
        else:
            header += "}"

        output.append(f"\\begin{{deluxetable}}{col_def}")
        output.append(f"\\tablecaption{{Polynomial Coefficients for {self.ext_fit_filter} Fit}}")
        output.append(header)
        output.append("\\startdata")

        idx = 0
        # 2. AKs terms (Group 'a')
        for p in range(self.order + 1):
            var_math = f"{aks}^{{{p}}}" if p > 0 else "1"
            coeff_id = f"a_{{{p}}}"
            row = f"${coeff_id}$ & ${var_math}$ & {coeffs[idx]:.{precision}f}"
            if include_errors:
                row += f" & {errors[idx]:.{precision}f}"
            output.append(row + " \\\\")
            idx += 1

        # 3. Color cross terms (Groups 'b', 'c'...)
        for i in range(len(self.ext_fit_colors)):
            ci_base = f"C_{{{i+1}}}"
            letter = letters[i + 1]
            sub_idx = 0
            for p in range(1, self.order + 1):
                for q in range(self.order + 1):
                    if p + q <= self.order:
                        c_part = f"{ci_base}" + (f"^{{{p}}}" if p > 1 else "")
                        a_part = f"{aks}" + (f"^{{{q}}}" if q > 1 else "") if q > 0 else ""
                        var_math = f"{c_part}{a_part}"
                        coeff_id = f"{letter}_{{{sub_idx}}}"
                        
                        row = f"${coeff_id}$ & ${var_math}$ & {coeffs[idx]:.{precision}f}"
                        if include_errors:
                            row += f" & {errors[idx]:.{precision}f}"
                        output.append(row + " \\\\")
                        idx += 1
                        sub_idx += 1
        
        output.append("\\enddata")

        # 4. Escape underscores in color names for the comment
        escaped_colors = [c.replace('_', '\\_') for c in self.ext_fit_colors]
        color_key = ", ".join([f"C_{{{i+1}}} = \\text{{{c}}}" for i, c in enumerate(escaped_colors)])
        
        output.append(f"\\tablecomments{{${color_key}$}}")
        output.append("\\end{deluxetable}")
        
        return "\n".join(output)

    def combine_deluxetables(self, table_strings, include_errors=True):
        """
        Combines multiple deluxetable strings side-by-side with localized color keys.
        
        An llm was used to generate the first draft of this function.
        """
        import re

        filter_data = {}
        filter_color_keys = {}
        all_term_ids = []
        term_to_var = {}

        for table in table_strings:
            filt_match = re.search(r"for (.*?) Fit", table)
            filt_name = filt_match.group(1) if filt_match else "Unknown"
            filter_data[filt_name] = {}
            
            # FIXED REGEX: Use non-greedy (.*?) and look ahead for the closing brace and newline
            # This prevents grabbing the subsequent \end{deluxetable}
            comment_match = re.search(r"\\tablecomments\{(.*?)\}\n\\end\{deluxetable\}", table, re.DOTALL)
            if comment_match:
                # Remove any existing $ signs so we can wrap them cleanly
                raw_comment = comment_match.group(1).replace('$', '').strip()
                filter_color_keys[filt_name] = raw_comment
            
            data_match = re.search(r"\\startdata\n(.*?)\n\\enddata", table, re.DOTALL)
            if data_match:
                rows = data_match.group(1).strip().split('\\\\')
                for row in rows:
                    if '&' not in row: continue
                    cols = [c.strip() for c in row.split('&')]
                    
                    term_id = cols[0]
                    var_label = cols[1]
                    
                    val = cols[2]
                    if include_errors and len(cols) > 3:
                        err = cols[3]
                        val_str = "$" + val + " \\pm " + err + "$"
                    else:
                        val_str = "$" + val + "$"
                    
                    filter_data[filt_name][term_id] = val_str
                    if term_id not in all_term_ids:
                        all_term_ids.append(term_id)
                        term_to_var[term_id] = var_label

        # 1. Build Headers
        n_filters = len(filter_data)
        col_def = "lc" + "c" * n_filters
        
        header_parts = ["\\colhead{ID}", "\\colhead{Term}"]
        for f in filter_data.keys():
            clean_f = f.replace('_', '\\_')
            header_parts.append("\\colhead{" + clean_f + "}")
        
        combined = [
            "\\begin{deluxetable}{" + col_def + "}",
            "\\tablecaption{Extinction coefficient fit results}",
            "\\tablehead{" + " & ".join(header_parts) + "}",
            "\\startdata"
        ]

        # 2. Build Rows
        for tid in all_term_ids:
            row_str = tid + " & " + term_to_var[tid]
            for filt_name in filter_data.keys():
                val = filter_data[filt_name].get(tid, " - ")
                row_str += " & " + val
            combined.append(row_str + " \\\\")

        combined.append("\\enddata")
        
        # 3. Build Legend (CLEANED)
        legend_entries = []
        for filt, colors in filter_color_keys.items():
            clean_filt = filt.replace('_', '\\_')
            # Each filter gets its own bold name and its own math block
            legend_entries.append("\\textbf{" + clean_filt + "}: $" + colors + "$")
        
        # Join with semicolons and wrap in the final tablecomments
        combined.append("\\tablecomments{" + "; ".join(legend_entries) + "}")
        combined.append("\\end{deluxetable}")
        
        return "\n".join(combined)


    import string

    @staticmethod
    def print_all_coeffs_deluxetable(data_dict, precision=5):
        """
        Generates a single multi-column LaTeX deluxetable mapping coefficients
        exactly as they are unpacked by generic_extinction_polynomial.
        Places color term labels as descriptive data rows at the top of the table.
        Generically extracts the last part of any underscore-separated string for display.
        Removes trailing '-abs' suffixes from the color terms.
        Uses coefficient IDs matching the formal triple-sum equation notation:
          - Pure AKs terms: a_{i}
          - Cross-terms:    b_{k,j,i} (color k, color power j, AKs power i)
        Safe for all Python versions (no backslashes inside f-string expressions).
        """
        if not data_dict:
            return "% Empty data dictionary provided."
            
        filters = list(data_dict.keys())
        num_filters = len(filters)
        
        # 1. Establish the global maximums to size our table rows correctly
        max_order = max(content["order"] for content in data_dict.values())
        
        # Helper function to reliably flatten any nested color structures
        def flatten_colors(f_colors):
            flat = []
            if isinstance(f_colors, list):
                for item in f_colors:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(item)
            else:
                flat.append(f_colors)
            return flat

        # Generic helper function to split by underscore and grab only the last item
        def generic_clean_name(text):
            if not text:
                return ""
            text_str = str(text)
            # Strip trailing -abs suffix before parsing elements
            if text_str.endswith("-abs"):
                text_str = text_str[:-4]
            parts = text_str.split('_')
            last_part = parts[-1] if parts else text_str
            return last_part.replace('_', '\\_')

        # Find the maximum number of colors any single filter uses based on the flattened structure
        max_colors = 0
        for f in filters:
            flat_f_colors = flatten_colors(data_dict[f]["colors"])
            max_colors = max(max_colors, len(flat_f_colors))

        # 2. Deconstruct the coefficient list for EACH filter matching generic_extinction_polynomial
        aks = "A_{Ks}"
        
        # Store matrix layout: {(term_type, color_idx, p, q): {filter: value}}
        master_table_rows = {}
        
        # Pre-populate pure AKs terms
        for p in range(max_order + 1):
            master_table_rows[('aks', 0, p, 0)] = {}
            
        # Pre-populate cross terms grouped by local color index
        for c_idx in range(max_colors):
            for p in range(1, max_order + 1):
                for q in range(max_order + 1):
                    if p + q <= max_order:
                        master_table_rows[('cross', c_idx, p, q)] = {}

        # Map the dictionary flat arrays into the structural math slots
        for f in filters:
            coeffs = data_dict[f]["coefficients"]
            order = data_dict[f]["order"]
            
            flat_f_colors = flatten_colors(data_dict[f]["colors"])
            n_colors = len(flat_f_colors)
            idx = 0
            
            # Unpack pure AKs terms
            for p in range(order + 1):
                if idx < len(coeffs):
                    master_table_rows[('aks', 0, p, 0)][f] = coeffs[idx]
                    idx += 1
                    
            # Unpack cross terms matching the nested loops of your polynomial
            for c_idx in range(n_colors):
                for p in range(1, order + 1):
                    for q in range(order + 1):
                        if p + q <= order:
                            if idx < len(coeffs):
                                master_table_rows[('cross', c_idx, p, q)][f] = coeffs[idx]
                                idx += 1

        # 3. Assemble the LaTeX string output
        col_def = "lc" + "c" * num_filters
        
        header_cells = ["\\colhead{ID}", "\\colhead{Term}"]
        for f in filters:
            display_filter_name = generic_clean_name(f)
            header_cells.append("\\colhead{" + display_filter_name + "}")
            
        header = "\\tablehead{" + " & ".join(header_cells) + "}"
        
        output = []
        output.append("\\begin{deluxetable}{" + col_def + "}")
        output.append("\\tablecaption{Polynomial Coefficients Summary Fit}")
        output.append(header)
        output.append("\\startdata")
        
        # --- Inject Metadata Rows at the Top ---
        for c_idx in range(max_colors):
            row_id = "$C_{" + str(c_idx + 1) + "}$"
            row_term = " "
            meta_cells = [row_id, row_term]
            
            for f in filters:
                flat_f_colors = flatten_colors(data_dict[f]["colors"])
                if c_idx < len(flat_f_colors):
                    raw_color = flat_f_colors[c_idx]
                    # Safely handle the removal of -abs if it sits at the very end of the full color entry
                    if raw_color.endswith("-abs"):
                        raw_color = raw_color[:-4]
                    color_components = raw_color.split('-')
                    cleaned_components = [generic_clean_name(comp) for comp in color_components]
                    display_color_name = "-".join(cleaned_components)
                    meta_cells.append(display_color_name)
                else:
                    meta_cells.append(r"\nodata")
            output.append(" & ".join(meta_cells) + " \\\\")
            
        output.append("\\hline")
        
        # Helper formatting specifier for precision
        float_fmt = "{:." + str(precision) + "f}"
        
        # 4. Generate the strings for each active row slot
        for (term_type, c_idx, p, q), filter_values in master_table_rows.items():
            if not filter_values:
                continue
                
            if term_type == 'aks':
                coeff_id = "a_{" + str(p) + "}"
                var_math = f"{aks}^{{{p}}}" if p > 0 else "1"
            else:
                # Local color variable index mapping
                ci_base = "C_{" + str(c_idx + 1) + "}"
                
                # Equation Mapping Rules:
                # k = c_idx + 1 (color index starting at 1)
                # j = p         (power of the color term)
                # i = q         (power of the AKs term)
                k_val = c_idx + 1
                j_val = p
                i_val = q
                
                coeff_id = "b_{" + str(k_val) + "," + str(j_val) + "," + str(i_val) + "}"
                
                c_part = f"{ci_base}" + (f"^{{{p}}}" if p > 1 else "")
                a_part = f"{aks}" + (f"^{{{q}}}" if q > 1 else "") if q > 0 else ""
                var_math = f"{c_part}{a_part}"
                
            row_cells = [f"${coeff_id}$", f"${var_math}$"]
            
            for f in filters:
                if f in filter_values:
                    row_cells.append(float_fmt.format(filter_values[f]))
                else:
                    row_cells.append(r"\nodata")
                    
            output.append(" & ".join(row_cells) + " \\\\")
            
        output.append("\\enddata")
        output.append("\\end{deluxetable}")
        
        return "\n".join(output)

    
    @staticmethod
    def save_all_coeffs_machine_readable(data_dict, fname, precision=5):
        columns = ['Filter']
        max_n_colors = np.max([len(fit['colors']) for fit in data_dict.values()])
        orders = [fit['order'] for fit in data_dict.values()]
        assert len(set(orders)) <= 1, "this function requires equal order for all fits"
        max_order = orders[0]
        columns += [f'Color{i}' for i in range(1,max_n_colors+1)]
        columns += [f'a_{i}' for i in range(max_order+1)]
        for i in range(1, max_n_colors+1):
            for p in range(1, max_order+1):
                for q in range(max_order+1):
                    if p + q <= max_order:
                        columns.append(f'b_{i}{p}{q}')
        tab = Table(names=columns)
        for col in ['Filter'] + [f'Color{i}' for i in range(1,max_n_colors+1)]:
            tab.replace_column(col, tab[col].astype(str))
        vals_per_color = np.sum(range(max_order+1))
        for i, (key, value) in enumerate(data_dict.items()):
            row = [key] + value['colors'] + ['' for i in range(max_n_colors-len(value['colors']))]
            row += value['coefficients']
            for x in range(max_n_colors-len(value['colors'])):
                row += [999.0 for i in range(vals_per_color)]
            tab.add_row(row)
        for col in tab.columns:
            if (col.startswith('b_') and not col.startswith('b_1')) \
               or (col[:-1]=='Color' and not col=='Color1'):
                tab[col] = MaskedColumn(tab[col].data, name=col, mask=(tab[col]==999.0))
                
            #tab[col].description = 
          
        tab.write(fname, format='ascii.mrt', overwrite=True)
        return tab
    
    
    
    def plot_fit_result(self):
        if not hasattr(self, 'best_fit_function'):
            raise RuntimeError("A fit must be run before the results can be plotted.")
        fig,ax = plt.subplots(nrows=1, ncols=len(self.loggs), sharey=True, 
                              figsize=(5*len(self.loggs),5), layout='constrained')

        for i in range(len(self.loggs)):
            idxs = self.ext_grid['logg']==self.loggs[i]
            im1 = ax[i].scatter(self.ext_grid['Teff'][idxs], self.ext_ests[idxs]-self.filt_ext_arr[idxs], 
                    c=self.ext_grid['A_Ks'][idxs], s=5)
            ax[i].set_title(f'logg={self.loggs[i]}')
            ax[i].set_xlabel('Teff (K)')
            ax[i].set_xscale('log')
            
        print(self.ext_fit_filter, np.max(np.abs(self.ext_ests/self.ext_grid['A_Ks']
                                                 - self.filt_ext_arr/self.ext_grid['A_Ks'])/
                                          (self.filt_ext_arr/self.ext_grid['A_Ks'])))

        fig.colorbar(im1, ax=ax[1], label='AKs')
        ax[0].set_ylabel(fr'$A_{{{self.ext_fit_filter.split("_")[-1]},fit}}'
                         fr'- A_{{{self.ext_fit_filter.split("_")[-1]},true}}$')
        fig.savefig(f'{self.figure_dir}ext_corr_grid_{self.ext_fit_filter}.png')
        return fig,ax

    def get_catalog_true(self, catalog):
        cat = catalog.copy()
        for i in range(len(self.color_filter_list[0])-1):
            col = f'{self.filters_short[i]}_{self.filters_short[i+1]}_abs_syn'
            f1, f2 = self.filter_synthpop_columns[i], self.filter_synthpop_columns[i+1]
            if (f1 in cat.columns) and (f2 in cat.columns):
                cat.loc[:,col] = cat[f1]-cat[f2]

        cat.loc[:,'Teff'] = 10**cat['log_Teff']
        for filt in self.filters_short:
            cat.loc[:,f'ext_{filt}_true'] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                for j in tqdm.tqdm(cat.index):
                    spec_base = self.atm_func(metallicity=cat['Fe/H_initial'][j],
                                         temperature=cat['Teff'][j], 
                                         gravity=cat['log_g'][j], verbose=False)
                    spec = copy.deepcopy(spec_base)  # in erg s^-1 cm^-2 A^-1
                    red = self.red_law.extinction_curve(cat['A_Ks'][j], spec.waveset)
                    spec *= red
                    mag_base = {}
                    mag = {}
                    for i,f in enumerate(self.filters_short[:len(self.color_filter_list[0])]):
                        mag_base[f] = synthetic.mag_in_filter(spec_base, self.filter_objs[i]) + self.mag_ab_vega[i]
                        mag[f] = synthetic.mag_in_filter(spec, self.filter_objs[i]) + self.mag_ab_vega[i]
                        cat.loc[j, f'ext_{f}_true'] = mag[f]-mag_base[f]
                    for i in range(len(self.color_filter_list[0])-1):
                        f1,f2 = self.filters_short[i], self.filters_short[i+1]
                        c = f'{self.filters_short[i]}_{self.filters_short[i+1]}'
                        cat.loc[j,c+'_app_spi'] = (mag[f1]-mag[f2])
                        cat.loc[j,c+'_abs_spi'] = (mag_base[f1]-mag_base[f2])
        return cat
    
    def plot_catalog_results(self, cat, maglim=None, ext='', use_syn=False,
                             use_app=False):
        ext_fit_colors = ['_'.join(c.split('-')) for c in self.ext_fit_colors]
        if use_syn:
            cat.loc[:,f'ext_{self.ext_fit_filter}_fit'] = self.best_fit_function(cat[['A_Ks', 
                        ]+[c+'_syn' for c in ext_fit_colors]].to_numpy())
        elif not use_app:
            cat.loc[:,f'ext_{self.ext_fit_filter}_fit'] = self.best_fit_function(cat[['A_Ks', 
                        ]+[c+'_spi' for c in ext_fit_colors]].to_numpy())
        elif use_app:
            cat.loc[:,f'ext_{self.ext_fit_filter}_fit'] = self.best_fit_function(cat[['A_Ks', 
                        ]+[c+'_app_spi' for c in ext_fit_colors]].to_numpy())
        cat_obs_mag = cat[self.filter_synthpop_columns[self.filters_short.index(self.ext_fit_filter)]] \
                        + 5*np.log10(100*cat['Dist']) + cat[f'ext_{self.ext_fit_filter}_fit']
        idxs = cat.index
        if maglim is not None:
            idxs = cat_obs_mag<maglim
        
        plt.axhline(0, c='k')
        plt.scatter(cat['Teff'][idxs], cat[f'ext_{self.ext_fit_filter}_fit'][idxs] *cat['A_Ks'][idxs]  - \
                    cat[f'ext_{self.ext_fit_filter}_true'][idxs], c=cat['A_Ks'][idxs],
                    vmin=0,s=1, rasterized=True)
        plt.colorbar(label='A_Ks')
        plt.ylabel(fr'$A_{{{self.ext_fit_filter.split("_")[-1]}, fit}}$'+
                   fr'- $A_{{{self.ext_fit_filter.split("_")[-1]}, true}}$')
        plt.xlabel('Teff (K)')
        plt.tight_layout()
        if use_syn:
            ext += '_syn'
        plt.savefig(f'{self.figure_dir}ext_cat_test_{self.ext_fit_filter}{ext}.png')
        return plt
