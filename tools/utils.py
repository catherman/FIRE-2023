
# Sample utils from phase 1; need to modify for FIRE II
from datetime import date
import os
import pathlib

import pandas as pd
import yaml

from config import Constants, logger

# For some continuity in Phase 2, this is reprointed from Phase 1, code by https://github.com/atlytics/FireExplorer_Learner/blob/main/tools/utils.py

def read_file(relative_path: str, target_dir: str = Constants.PROJECT_DIR,
              header: list = 'infer', skiprows: [list, int] = None):
    """Interpret relative path and read file type to pandas dataframe.

    Args:
        relative_path (str): Relative path to the local read location.
        target_dir (str): Directory to search for the relative path.
        header (list): Row number(s) to use as column names and start of data.
        skiprows (list): Line number(s) to skip or number of lines to skip.
    """
    logger.info(f'[READ  ] {relative_path}')
    absolute_path = os.path.join(target_dir, relative_path)
    extension = pathlib.Path(absolute_path).suffix
    match extension:
        case '.csv':
            return pd.read_csv(absolute_path, header=header, skiprows=skiprows)
        case '.txt':
            with open(absolute_path, 'r') as f:
                text = f.read().split('\n')
                return text
        case '.yaml':
            with open(absolute_path) as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        case _:
            raise KeyError(f'Unsupported file type "{extension}".')


def write_file(df: pd.DataFrame, relative_path: str,
               target_dir: str = Constants.PROJECT_DIR,
               do_versioned: bool = False):
    """Interpret relative path and read file type to pandas dataframe.

    Args:
        df (pd.DataFrame): Dataframe to write.
        relative_path (str): Relative path to the local read location.
        target_dir (str): Directory to search for the relative path.
        do_versioned (bool): Save a versioned copy of the file.
    """
    logger.info(f'[WRITE ] {relative_path}')
    absolute_path = os.path.join(target_dir, relative_path)
    extension = pathlib.Path(absolute_path).suffix
    match extension:
        case '.csv':
            df.to_csv(absolute_path, index=False)
            if do_versioned:
                today = date.today().strftime(Constants.DATETIME_FORMAT)
                return f"{os.path.splitext(absolute_path)[0]}--" \
                       f"{today}{os.path.splitext(absolute_path)[-1]}"
        case _:
            raise KeyError(f'Unsupported file type "{extension}".')