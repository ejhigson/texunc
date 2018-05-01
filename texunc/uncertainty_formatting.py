#!/usr/bin/env python
"""
Convert values and numerical uncertainties into strings of format

1.234(5) \\cdot 10^{-6}

where number in brackets is error on the final digit.
"""

import numpy as np


__all__ = ['latex_form']


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
    power = _get_power(value_in, max_power, min_power)
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
    dp = int(_get_dp(error, min_dp))
    # make output
    output = '{:,.{prec}f}'.format(value, prec=dp)
    if (error is not None) and ~np.isnan(error):
        error *= (10 ** dp)
        output += '({:.{prec}f})'.format(error, prec=0)
    if power != 0:
        output = r'$' + output + r'\cdot 10^{' + str(power) + '}$'
    return output


# Helper functions
# ----------------


def _get_power(value_in, max_power, min_power):
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


def _get_dp(error, dp_min):
    """
    Find how many decimal places should be shown given the size of the
    numberical uncertainty.

    This is the number of decimal places which is needed to have the
    error on the final digit (rounded to 1 sigificant figure) in the range 1 to
    9. If this number is less than dp_min, dp_min is used.

    Parameters
    ----------
    error: float
    dp_min: int, None or NaN

    Returns
    -------
    dp_to_use: int
    """
    if error is None or np.isnan(error) or error == 0:
        return dp_min
    if error >= 1:
        return dp_min
    # have a float error that is less than 1 in magnitude:
    # find dp needed for error to be >= to 1
    dp_given_error = int(np.ceil(abs(np.log10(error))))
    # Reduce dp by 1 when the error in brackets rounds up to 10 so it is
    # instead shown as 1
    if np.rint(error * (10 ** dp_given_error)) == 10:
        dp_given_error -= 1
    return max(dp_min, dp_given_error)
