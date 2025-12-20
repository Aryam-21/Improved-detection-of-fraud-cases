import pandas as pd
import pickle
def load_data(path):
    return pd.read_csv(path)


def load_pickle(file_path):
    """
    Load a pickled Pandas DataFrame or Series.

    Parameters:
    -----------
    file_path : str
        Path to the pickle file.

    Returns:
    --------
    pd.DataFrame or pd.Series
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError(f"Loaded object is not a Pandas DataFrame or Series. Got {type(data)} instead.")
    
    return data