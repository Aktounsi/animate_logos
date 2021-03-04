import shutil
from src.utils import logger


def delete_dir(directory):
    """
    This function deletes the directory which is given as a parameter.
    Args:
        directory (string or list): If given a string, delete the respective directory.
        If given a list of directories, delete all directories recursively.
    """
    if isinstance(directory, str):
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            logger.warning(f"Directory {directory} does not exist and won't be deleted")
    elif isinstance(directory, list):
        for dir_ in directory:
            delete_dir(dir_)
    else:
        logger.error(f"Parameter 'directory' should be of type string or list")
