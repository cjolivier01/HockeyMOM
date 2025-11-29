import sys
from pathlib import Path

# Get the path of the current file (__init__.py)
current_file_path = Path(__file__)

# Get the parent directory of the current file (cli directory)
cli_directory = current_file_path.parent

# Get the parent directory of the cli directory (hmlib directory)
hmlib_directory = cli_directory.parent

# Insert the hmlib directory at the beginning of sys.path
sys.path.insert(0, str(hmlib_directory))

