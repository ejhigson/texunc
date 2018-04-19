#!/usr/bin/env python
"""
Convert values and numerical uncertainties into strings of format

1.234(5) \\cdot 10^{-6}

where number in brackets is error on the final digit.
"""

import numpy as np


def latex_form(value_in, error_in, **kwargs):
    """
    Convert value and error to form
    1.234(5) \\cdot 10^{-6}
    where number in brackets is error on the final digit.
    """
    max_power = kwargs.pop('max_power', 4)
    min_power = kwargs.pop('min_power', -max_power)
    min_dp = kwargs.pop('min_dp', 1)
    min_dp_no_error = kwargs.pop('min_dp_no_error', min_dp)
    zero_dp_ints = kwargs.pop('zero_dp_ints', True)
    if value_in is None or np.isnan(value_in):
        return str(value_in) + '(' + str(error_in) + ')'
    # Work out power and adjust error and values
    power = get_power(value_in, max_power, min_power)
    value = value_in / (10 ** power)
    if error_in is None or np.isnan(error_in):
        error = error_in
        if power == 0 and value_in == np.rint(value_in) and zero_dp_ints:
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
    star_table = kwargs.pop('star_table', False)
    caption = kwargs.pop('caption', 'Caption here.')
    label = kwargs.pop('label', 'tab:tbc')
    # Get latex df as string
    df = latex_format_df(df, **kwargs)
    # stop to_latex adding row for index names
    df.index.names = [None] * len(df.index.names)
    df_str = df.to_latex(escape=False)
    # format all the strings we need
    if str_map is not None:
        for key, value in str_map.items():
            df_str = df_str.replace(key, value)
    table_str = r'table'
    if star_table:
        table_str += r'*'
    caption_str = r'\caption{' + caption + r'}\label{' + label + r'}'
    # do the printing
    # ---------------
    print()  # print a new line
    print(r'\begin{' + table_str + '}')
    print(r'\centering')
    if caption_above:
        print(caption_str)
        print(df_str + r'\end{table*}')
    else:
        print(df_str + caption_str)
        print(r'\end{' + table_str + '}')
    return df


# Helper functions
# ----------------


def get_power(value_in, max_power, min_power):
    """
    Find power to use in standard form (if any).
    """
    if value_in == 0 or value_in == np.inf:
        return 0
    else:
        power = int(np.floor(np.log10(abs(value_in))))
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
