import pandas as pd

def moordyn_out(filepath):
    """Reads in MoorDyn OUT files such as "FAST.MD.out" and "FAST.MD.Line1.out".

    Parameters
    ----------
    filepath : str
        Path to file

    Returns
    -------
    pd.DataFrame
        Dataframe containing parsed MoorDyn OUT file
    """
    assert isinstance(filepath, str), 'filepath must be of type str'

    data = pd.read_csv(filepath, header=0, skiprows=[1], sep=' ', skipinitialspace=True, index_col=0)
    data = data.dropna(axis=1)

    return data


