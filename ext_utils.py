from spisea import synthetic, atmospheres, reddening
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os.path
import copy
import pandas as pd
import tqdm
import string
import pdb
import re
plt.rcParams['font.size'] = 14

# Some simple filter functions for convenience
def get_eff_lam(flt_name):
    import pysynphot
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
red_law_default = reddening.RedLawSODC(Rv=2.5)


class ExtinctionCoefficientFitter():

    def __init__(self, color_filter_list=['roman,wfi,f062','roman,wfi,f087','roman,wfi,f106','roman,wfi,f129',
                             'roman,wfi,f158','roman,wfi,f184','roman,wfi,f213','roman,wfi,f146'],
                       ext_only_filter_list=[],
                       AKs_grid = AKs_grid_default,
                       metallicity=0.0, loggs=[4.5,2], grid_dir='./grids', recompute_grid=False,
                       filter_synthpop_columns=["R062", "Z087", "Y106", "J129", "W146", "H158", None, "F184"],
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
        self.ext_grid.loc[:,'neg_A_Ks'] = -self.ext_grid['A_Ks']
        self.ext_grid.sort_values(by=['neg_A_Ks', 'logg', 'Teff'], inplace=True)
        self.ext_grid.drop(columns=['neg_A_Ks'], inplace=True)
        self.ext_grid.reset_index(inplace=True, drop=True)
        
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
        for teff in tqdm.tqdm(np.logspace(np.log10(2_500), np.log10(10_000), 100)):
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
            ax[i].set_title(f'A_Ks = {AKs:.2f}')
            ax[i].set_xticks([3e3,1e4,2e4])
            ax[i].set_xlabel('Teff (K)')
            ax[i].set_xscale('log')
            lam = get_eff_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks = getattr(self.red_law, self.red_law.name.split(',')[0])(lam/1e4, AKs)[0]
            textstr = r'$A_{\lambda,eff}$ = ' + f'{alam_aks:.3f}'
            lam = get_piv_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks = getattr(self.red_law, self.red_law.name.split(',')[0])(lam/1e4, AKs)[0]
            textstr += '\n'+r'$A_{\lambda,piv}$ = ' + f'{alam_aks:.3f}'
            lam = get_avg_lam(self.filters_long[self.filters_short.index(filt)])
            alam_aks = getattr(self.red_law, self.red_law.name.split(',')[0])(lam/1e4, AKs)[0]
            textstr += '\n'+r'$A_{\lambda,avg}$ = ' + f'{alam_aks:.3f}'
            ax[i].text(0.05, 0.95, textstr, transform=ax[i].transAxes,
                    verticalalignment='top')
        ax[0].legend(loc=4)
        ax[0].set_ylabel(r'$\Delta$'+filt+r'$_{true}$')
        
        os.makedirs('figures',exist_ok=True)
        print("saving figure")
        fig.savefig(f'{self.figure_dir}/ext_true_{filt}.png')
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
    
    def get_fit_dict(self):
        return {self.ext_fit_filter: {"colors": ['_'.join(c.split('_')[:2]) for c in self.ext_fit_colors],
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

        fig.colorbar(im1, ax=ax[1], label='AKs')
        ax[0].set_ylabel(r'$\Delta$'+self.ext_fit_filter+r'$_{\rm fit}$'+
                         r'- $\Delta$'+self.ext_fit_filter+r'$_{\rm true}$')
        fig.savefig(f'{self.figure_dir}ext_corr_grid_{self.ext_fit_filter}.png')
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
        for j in tqdm.tqdm(cat.index):
            spec_base = self.atm_func(metallicity=cat['Fe/H_initial'][j],
                                 temperature=cat['Teff'][j], 
                                 gravity=cat['log_g'][j], verbose=False)
            spec = copy.deepcopy(spec_base)  # in erg s^-1 cm^-2 A^-1
            red = self.red_law.reddening(cat['A_Ks'][j]).resample(spec.wave)
            spec *= red
            mag_base = {}
            mag = {}
            for i,f in enumerate(self.filters_short):
                mag_base[f] = synthetic.mag_in_filter(spec_base, self.filter_objs[i]) + self.mag_ab_vega[i]
                mag[f] = synthetic.mag_in_filter(spec, self.filter_objs[i]) + self.mag_ab_vega[i]
                cat.loc[j, f'ext_{f}_true'] = mag[f]-mag_base[f]
            for i in range(len(self.color_filter_list)-1):
                f1,f2 = self.filters_short[i], self.filters_short[i+1]
                c = f'{self.filters_short[i]}_{self.filters_short[i+1]}'
                cat.loc[j,c+'_app_spi'] = (mag[f1]-mag[f2])
                cat.loc[j,c+'_abs_spi'] = (mag_base[f1]-mag_base[f2])
        return cat
    
    def plot_catalog_results(self, cat, maglim=None, ext=''):
        cols = [f'{self.filters_short[i]}_{self.filters_short[i+1]}_abs_syn' 
                for i in range(len(self.color_filter_list)-1)]
        try:
            cat.loc[:,f'ext_{self.ext_fit_filter}_fit'] = self.best_fit_function(cat[['A_Ks', 
                        ]+[c+'_syn' for c in self.ext_fit_colors]].to_numpy())
        except:
            cat.loc[:,f'ext_{self.ext_fit_filter}_fit'] = self.best_fit_function(cat[['A_Ks', 
                        ]+[c+'_app_spi' for c in self.ext_fit_colors]].to_numpy())
        cat_obs_mag = cat[self.filter_synthpop_columns[self.filters_short.index(self.ext_fit_filter)]] \
                        + 5*np.log10(100*cat['Dist']) + cat[f'ext_{self.ext_fit_filter}_fit']
        idxs = cat.index
        if maglim is not None:
            idxs = cat_obs_mag<maglim
        
        plt.axhline(0, c='k')
        plt.scatter(cat['Teff'][idxs], cat[f'ext_{self.ext_fit_filter}_fit'][idxs] - \
                    cat[f'ext_{self.ext_fit_filter}_true'][idxs], c=cat['A_Ks'][idxs],
                    vmin=0,s=1)
        plt.colorbar(label='A_Ks')
        plt.ylabel(r'$\Delta$'+self.ext_fit_filter+r'$_{\rm fit}$'+
                   r'- $\Delta$'+self.ext_fit_filter+r'$_{\rm true}$')
        plt.xlabel('Teff (K)')
        plt.tight_layout()
        plt.savefig(f'{self.figure_dir}ext_cat_test_{self.ext_fit_filter}{ext}.png')
        return plt
