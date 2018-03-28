#!/usr/bin/env python
"""
Convert values and numerical uncertainties into strings of format

1.234(5) \\cdot 10^{-6}


where number in brackets is error on the final digit.
"""

import copy
import numpy as np


def latex_form(value_in, error_in, **kwargs):
    """
    Convert value and error to form
    1.234(5) \\cdot 10^{-6}
    where number in brackets is error on the final digit.
    """
    max_power = kwargs.pop('max_power', 4)
    min_power = kwargs.pop('min_power', -4)
    min_dp = kwargs.pop('min_dp', 1)
    min_dp_no_error = kwargs.pop('min_dp_no_error', 1)
    if value_in is None or np.isnan(value_in):
        return str(value_in) + '(' + str(error_in) + ')'
    # Work out power and adjust error and values
    power = get_power(value_in, max_power, min_power)
    value = value_in / (10 ** power)
    if error_in is None or np.isnan(error_in):
        error = error_in
        if power == 0 and value_in == np.rint(value_in):
            min_dp = 0
        else:
            min_dp = min_dp_no_error
    else:
        error = error_in / (10 ** power)
    # Work out decimal places
    dp = int(get_dp(error, min_dp))
    # make output
    output = '{:,.{prec}f}'.format(value, prec=dp)
    if (error is not None) and ~np.isnan(error):
        error *= (10 ** dp)
        output += '({:.{prec}f})'.format(error, prec=0)
    if power != 0:
        output = r'$' + output + r'\cdot 10^{' + str(power) + '}$'
    return output


def latex_format_df(df, **kwargs):
    """
    Format a pandas multiindex DataFrame with index 'result type' containing
    'value' and 'uncertainty' using latex_form.
    """
    assert 'result type' in df.index.names
    # Save the order to make sure we don't change it
    order = df.index.droplevel('result type').unique()
    latex_df = (df.stack().unstack(level='result type')
                .apply(pandas_latex_form_apply, axis=1, **kwargs).unstack())
    return latex_df.reindex(order)


def print_latex_df(df, str_map=None, caption_above=True, **kwargs):
    """
    Formats df and prints it out (can copy paste into tex file).
    """
    if str_map is None:
        str_map = {'mathrm{log}': 'log',
                   'implementation': 'Implementation',
                   'efficiency gain': 'Efficiency gain',
                   'std': r'\std{}',
                   'None': '',
                   '.0000': ''}
    df = latex_format_df(df, **kwargs)
    df_str = df.to_latex(escape=False)
    for key, value in str_map.items():
        df_str = df_str.replace(key, value)
    print()  # print a new line
    print(r'\begin{table*}')
    print(r'\centering')
    if caption_above:
        print(r'\caption{Caption here.}\label{tab:tbc}')
        print(df_str + r'\end{table*}')
    else:
        print(df_str + r'\caption{Caption here.}\label{tab:tbc}')
        print(r'\end{table*}')


def paper_eff_df(eff_df):
    """
    Transform efficiency gain data frames output by nestcheck into the format
    used in the dns paper.
    """
    row_name_map = {'std efficiency gain': 'Efficiency gain',
                    'dynamic ': ''}
    comb_df = copy.deepcopy(eff_df)
    comb_df = comb_df.loc[comb_df.index.get_level_values(0) != 'mean']
    # Show mean number of samples and likelihood calls instead of st dev
    means = (eff_df.xs('mean', level='calculation type')
             .xs('value', level='result type'))
    for col in ['samples', 'likelihood calls']:
        try:
            col_vals = []
            for val in means[col].values:
                col_vals += [int(np.rint(val)), np.nan]
            col_vals += [np.nan] * (comb_df.shape[0] - len(col_vals))
            comb_df[col] = col_vals
        except KeyError:
            pass
    row_names = (comb_df.index.get_level_values(0).astype(str) + ' ' +
                 comb_df.index.get_level_values(1).astype(str))
    for key, value in row_name_map.items():
        row_names = row_names.str.replace(key, value)
    comb_df.index = [row_names, comb_df.index.get_level_values(2)]
    return comb_df

# Helper functions
# ----------------


def get_power(value_in, max_power, min_power):
    """
    Find power to use in standard form (if any).
    """
    if value_in == 0 or value_in == np.inf:
        return 0
    else:
        power = int(np.log10(abs(value_in)))
    if max_power >= power >= min_power:
        return 0
    else:
        return power


def get_dp(error, dp_min):
    """
    Find how many decimal places should be shown given the size of the
    numberical uncertainty.
    """
    if error is None:
        return dp_min
    elif np.isnan(error):
        return dp_min
    elif error == 0:
        return dp_min
    elif error >= 1:
        return dp_min
    else:
        # have a float error that is less than 1 in magnitude:
        # find dp needed for error to be >= to 1
        dp_given_error = int(np.ceil(abs(np.log10(error))))
        # Reduce dp by 1 when the error in brackets rounds up to 10 so it is
        # instead shown as 1
        if np.rint(error * (10 ** dp_given_error)) == 10:
            dp_given_error -= 1
        return max(dp_min, dp_given_error)


def pandas_latex_form_apply(series, **kwargs):
    """
    Helper function for applying latex_form to pandas multiindex.
    """
    # assert np.all(series.index.categories == pd.Index(['value',
    #                                                    'uncertainty']))
    assert series.shape == (1,) or series.shape == (2,)
    if series.shape == (1,):
        assert series.index.values[0] == 'value'
        str_out = latex_form(series.loc['value'], None, **kwargs)
    if series.shape == (2,):
        assert np.all(series.index.values == ['value', 'uncertainty'])
        str_out = latex_form(series.loc['value'], series.loc['uncertainty'],
                             **kwargs)
    return str_out
