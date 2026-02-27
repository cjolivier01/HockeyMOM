import os
import re
from pathlib import Path
from typing import Optional, Union


def sanitize_game_id_for_filename(game_id: Optional[str]) -> str:
    """Sanitize a game_id so it is safe to embed in a single filename."""
    if game_id is None:
        return ""
    value = str(game_id).strip()
    if not value:
        return ""
    # Replace any run of path separators with a single underscore.
    return re.sub(r"[\\/]+", "_", value)


def add_game_id_prefix_to_filename(
    path: Union[Path, str], game_id: Optional[str], sep: str = "-"
) -> Union[Path, str]:
    """Prefix a filename with a sanitized game_id (useful for deploy dirs)."""
    return add_prefix_to_filename(path, sanitize_game_id_for_filename(game_id), sep=sep)


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


def add_prefix_to_filename(
    path: Union[Path, str], prefix: Optional[str], sep: str = "_"
) -> Union[Path, str]:
    """
    Adds a prefix to the filename in a given path using pathlib, keeping the directory intact.

    Args:
    path (str or Path): The original file path.
    prefix (str): The prefix to add to the filename.
    sep (str): Separator inserted between prefix and filename.

    Returns:
    Path: The modified file path with the prefix added before the filename.
    """
    if prefix is None:
        return path
    prefix_str = str(prefix).strip()
    if not prefix_str:
        return path
    is_path = isinstance(path, Path)
    path = Path(path)
    name = path.name
    effective_sep = "" if prefix_str.endswith(("_", "-")) else sep
    if name.startswith(prefix_str):
        if len(name) == len(prefix_str):
            return path if is_path else str(path)
        if name[len(prefix_str)] in ("_", "-"):
            return path if is_path else str(path)
        if effective_sep and name.startswith(f"{prefix_str}{effective_sep}"):
            return path if is_path else str(path)
    new_name = f"{prefix_str}{effective_sep}{name}"
    new_path = path.with_name(new_name)
    return new_path if is_path else str(new_path)


def is_second_file_older(first: Union[str, Path], second: Union[str, Path]):
    """
    Takes two file names and returns True if the first file is older than the second file.
    """
    # Get the last modified time for both files
    first_mtime = os.path.getmtime(first)
    second_mtime = os.path.getmtime(second)

    # Compare the modification times
    return first_mtime < second_mtime
