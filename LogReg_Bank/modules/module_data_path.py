from pathlib import Path


def csv_data_path() -> Path:
    """
    Returns the location of the Banking CSV data, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the CSV file
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data"
        if data_folder.exists() and data_folder.is_dir():
            print("Data directory found in ", data_folder)
            return data_folder / "banking.csv"
        else:
            raise Exception("Data not found")
        
def model_data_path() -> Path:
    """
    Returns the location of the Banking CSV data, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the CSV file
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "models"
        if data_folder.exists() and data_folder.is_dir():
            print("Models directory found in ", data_folder)
            return data_folder
        else:
            raise Exception("Data not found")
        
def mlruns_data_path() -> Path:
    """
    Returns the location of the Banking CSV data, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the CSV file
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "mlruns"
        if data_folder.exists() and data_folder.is_dir():
            print("Mlruns directory found in ", data_folder)
            return data_folder
        else:
            raise Exception("Data not found")