"""
Functions to save and load DGSA results.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Union, Optional

def DGSA_save(dgsa_results: Dict, 
              filepath: Union[str, Path], 
              use_numpy: bool = False) -> None:
    """
    Save DGSA results to a file.

    Parameters
    ----------
    dgsa_results : Dict
        Dictionary containing DGSA results
    filepath : str or Path
        Path where to save the results. If use_numpy is True, '.npz' will be appended.
        If use_numpy is False, '.pkl' will be appended if no extension is provided.
    use_numpy : bool, optional
        If True, use numpy's savez_compressed. If False, use pickle (default).
        Numpy format is more space efficient but less flexible with Python objects.

    Returns
    -------
    None
    """
    filepath = Path(filepath)
    
    if use_numpy:
        if filepath.suffix != '.npz':
            filepath = filepath.with_suffix('.npz')
        np.savez_compressed(filepath, **dgsa_results)
    else:
        if not filepath.suffix:
            filepath = filepath.with_suffix('.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(dgsa_results, f)

def DGSA_load(filepath: Union[str, Path]) -> Dict:
    """
    Load DGSA results from a file.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved DGSA results file (.npz or .pkl)

    Returns
    -------
    Dict
        Dictionary containing DGSA results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        # Load numpy format
        with np.load(filepath, allow_pickle=True) as data:
            # Convert numpy arrays to dict
            results = {key: data[key] for key in data.files}
    else:
        # Load pickle format
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
    return results 