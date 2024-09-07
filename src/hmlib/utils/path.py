from pathlib import Path
from typing import Union


def add_suffix_to_filename(path: Union[Path, str], suffix: str) -> Union[Path, str]:
    """
    Adds a suffix to the filename in a given path using pathlib, keeping the directory and file extension unchanged.

    Args:
    path (str or Path): The original file path.
    suffix (str): The suffix to add to the filename.

    Returns:
    Path: The modified file path with the suffix added before the file extension.
    """
    is_path = isinstance(path, Path)
    # Convert path to a Path object if it's not already one
    path = Path(path)

    # Construct the new filename with the suffix
    new_filename = f"{path.stem}{suffix}{path.suffix}"

    # Return the new path
    new_path = path.with_name(new_filename)
    return new_path if not is_path else Path(new_path)
