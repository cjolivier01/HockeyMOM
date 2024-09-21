import time
from typing import Dict, List, Tuple

from hmlib.stitching.control_points import calculate_control_points

_CONTROL_POINTS_LINE = "# control points"


def load_pto_file(file_path: str) -> List[str]:
    """Load the content of a .pto file into a list of lines."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    # trim trailing whitespace
    for i, line in enumerate(lines):
        lines[i] = line.rstrip()
    return lines


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


def save_pto_file(file_path: str, data: List[str]):
    """Save modified data back to a .pto file."""
    with open(file_path, "w") as file:
        for line in data:
            file.write(f"{line}\n")


def remove_control_points(lines: List[str]) -> Tuple[List[str], int]:
    prev_control_point_count: int = 0
    new_lines: List[str] = []
    for line in lines:
        if line.startswith(_CONTROL_POINTS_LINE):
            continue
        if line.startswith("c "):
            prev_control_point_count += 1
            continue
        new_lines.append(line)

    return (new_lines, prev_control_point_count)


def configure_control_points(
    project_file_path: str, image0: str, image1: str, force: bool = False
) -> None:
    #  c n0 N1 x5162 y1173 X1416.1875 Y1252.78125 t0
    pto_file = load_pto_file(project_file_path)
    pto_file, prev_control_point_count = remove_control_points(lines=pto_file)
    if prev_control_point_count and not force:
        return
    start = time.time()
    control_points = calculate_control_points(image0=image0, image1=image1)
    print(f"Calculated control points in {time.time() - start} seconds")
    pts0 = control_points["m_kpts0"]
    pts1 = control_points["m_kpts1"]
    assert len(pts0) == len(pts1)
    print(f"Found {len(pts0)} control points")
    assert len(pts0) and len(pts1)
    pto_file.append("")
    pto_file.append(_CONTROL_POINTS_LINE)
    for i in range(len(pts0)):
        point0 = [float(c) for c in pts0[i]]
        point1 = [float(c) for c in pts1[i]]
        line = f"c n0 N1 x{float(point0[0])} y{float(point0[1])} X{float(point1[0])} Y{float(point1[1])} t0"
        pto_file.append(line)
    save_pto_file(file_path=project_file_path, data=pto_file)
    print("Done with control points")
