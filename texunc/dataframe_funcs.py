#!/usr/bin/env python
"""
Functions for applying uncertainty formatting to pandas DataFrames.
"""

from texunc.uncertainty_formatting import latex_form


__all__ = ['latex_format_df', 'print_latex_df']


def latex_format_df(df, **kwargs):
    """
    Format a pandas multiindex DataFrame with index 'result type' containing
    'value' and 'uncertainty' using latex_form.
    """
    assert 'result type' in df.index.names
    # Save the order to make sure we don't change it
    order = df.index.droplevel('result type').unique()
    latex_df = (df.stack().unstack(level='result type')
                .apply(_pandas_latex_form_apply, axis=1, **kwargs).unstack())
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
        print(df_str + r'\end{' + table_str + '}')
    else:
        print(df_str + caption_str)
        print(r'\end{' + table_str + '}')
    return df


# Helper functions
# ----------------


def _pandas_latex_form_apply(series, **kwargs):
    """
    Helper function for applying latex_form to pandas multiindex. This can be
    applied to each row and column except 'result type' to get the value and
    uncertainties as a string using latex_form. Works with some missing
    uncertainties.

    Series must be length 1 with index.values=['value'] or length 2 with
    index.values=['value', 'uncertainty'] (in any order). If it does not
    contain an uncertainty then None is used.

    Parameters
    ----------
    series: pandas Series
    kwargs: dict
        Keyword args for latex_form.

    Returns
    -------
    str_out: string
        See latex_form docstring for more details.
    """
    assert series.shape == (1,) or series.shape == (2,)
    if series.shape == (1,):
        assert series.index.values[0] == 'value'
        str_out = latex_form(series.loc['value'], None, **kwargs)
    if series.shape == (2,):
        inds = sorted(series.index.values)
        assert inds[0] == 'uncertainty' and inds[1] == 'value', (
            str(series.index.values))
        str_out = latex_form(series.loc['value'], series.loc['uncertainty'],
                             **kwargs)
    return str_out
