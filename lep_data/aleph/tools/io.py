import os
import glob
import tqdm
import awkward as ak


def load_parquet(input_path: str, columns: list = None) -> ak.Array:
    """Loads the contents of the .parquet file specified by the input_path

    Args:
        input_path : str
            The path to the .parquet file to be loaded.
        columns : list
            Names of the columns/branches to be loaded from the .parquet file

    Returns:
        input_data : ak.Array
            The data from the .parquet file
    """
    ret = ak.from_parquet(input_path, columns=columns)
    ret = ak.Array({k: ret[k] for k in ret.fields})
    return ret


def load_all_data(input_loc, n_files: int = None, columns: list = None) -> ak.Array:
    """Loads all .parquet files specified by the input. The input can be a list of input_paths, a directory where the files
    are located or a wildcard path.

    Args:
        input_loc : str
            Location of the .parquet files.
        n_files : int
            [default: None] Maximum number of input files to be loaded. By default all will be loaded.
        columns : list
            [default: None] Names of the columns/branches to be loaded from the .parquet file. By default all columns will
            be loaded

    Returns:
        input_data : ak.Array
            The concatenated data from all the loaded files
    """
    if n_files == -1:
        n_files = None
    if isinstance(input_loc, list):
        input_files = input_loc[:n_files]
    elif isinstance(input_loc, str):
        if os.path.isdir(input_loc):
            input_loc = os.path.expandvars(input_loc)
            input_files = glob.glob(os.path.join(input_loc, "*.parquet"))[:n_files]
        elif "*" in input_loc:
            input_files = glob.glob(input_loc)[:n_files]
        elif os.path.isfile(input_loc):
            input_files = [input_loc]
        else:
            raise ValueError(f"Unexpected input_loc")
    else:
        raise ValueError(f"Unexpected input_loc")
    input_data = []
    for file_path in tqdm.tqdm(sorted(input_files)):
        # for i, file_path in enumerate(sorted(input_files)):
        #     print(f"[{i+1}/{len(input_files)}] Loading from {file_path}")
        try:
            input_data.append(load_parquet(file_path, columns=columns))
        except ValueError:
            print(f"{file_path} does not exist")
    if len(input_data) > 0:
        data = ak.concatenate(input_data)
        print("Input data loaded")
    else:
        raise ValueError(f"No files found in {input_loc}")
    return data
