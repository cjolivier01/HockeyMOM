from typing import Dict, List

from hmlib.stitching.control_points import calculate_control_points


def load_pto_file(file_path: str) -> List[str]:
    """Load the content of a .pto file into a list of lines."""
    with open(file_path, "r") as file:
        return file.readlines()


def parse_pto_content(lines: List[str]) -> Dict[str, str]:
    """Parse the loaded .pto content to extract and possibly modify data."""
    parsed_data: Dict[str, str] = {}
    for line in lines:
        if line.startswith("#"):  # Skip comments
            continue
        key_value_pair = line.strip().split("=", 1)
        if len(key_value_pair) == 2:
            key, value = key_value_pair
            parsed_data[key] = value
    return parsed_data


def save_pto_file(file_path, data):
    """Save modified data back to a .pto file."""
    with open(file_path, "w") as file:
        for key, value in data.items():
            file.write(f"{key}={value}\n")


def configure_control_points(project_file_path: str, image0: str, image1: str) -> None:
    #  c n0 N1 x5162 y1173 X1416.1875 Y1252.78125 t0
    control_points = calculate_control_points(image0=image0, image1=image1)
    print("Done with control points")
