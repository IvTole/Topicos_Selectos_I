import pandas as pd
from typing import Optional, Tuple

def load_data_frame(path=None, num_samples:Optional[int]=None, random_seed: int = 42) -> pd.DataFrame:
    """"
    : param path: path of the csv file.
    : param num_samples: the number of samples to draw from the data frame; if None, use all samples.
    : param random_seed: the random seed to use when sampling data points
    """

    df = pd.read_csv(filepath_or_buffer=path)

    if num_samples is not None:
        df = df.sample(num_samples,
                       random_state=random_seed)
    print("Data is loaded")
    
    return df


