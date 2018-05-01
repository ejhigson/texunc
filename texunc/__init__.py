#!/usr/bin/env python
"""
Convert values and numerical uncertainties into strings of format

1.234(5) \\cdot 10^{-6}

where number in brackets is error on the final digit, and apply this to pandas
dataframes.
"""

from texunc.uncertainty_formatting import latex_form
from texunc.dataframe_funcs import latex_format_df, print_latex_df
