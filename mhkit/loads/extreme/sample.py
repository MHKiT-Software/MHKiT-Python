def return_year_value(ppf, return_year, short_term_period_hr):
    """
    Calculate the value from a given distribution corresponding to a particular
    return year.

    Parameters
    ----------
    ppf: callable function of 1 argument
        Percentage Point Function (inverse CDF) of short term distribution.
    return_year: int, float
        Return period in years.
    short_term_period_hr: int, float
        Short term period the distribution is created from in hours.

    Returns
    -------
    value: float
        The value corresponding to the return period from the distribution.
    """
    if not callable(ppf):
        raise TypeError("ppf must be a callable Percentage Point Function")
    if not isinstance(return_year, (float, int)):
        raise TypeError(
            f"return_year must be of type float or int. Got: {type(return_year)}"
        )
    if not isinstance(short_term_period_hr, (float, int)):
        raise TypeError(
            f"short_term_period_hr must be of type float or int. Got: {type(short_term_period_hr)}"
        )

    p = 1 / (return_year * 365.25 * 24 / short_term_period_hr)

    return ppf(1 - p)
