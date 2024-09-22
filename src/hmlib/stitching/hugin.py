import os
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import torch

from hmlib.stitching.control_points import calculate_control_points
from hmlib.utils.image import image_width

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


def strip(s: str) -> str:
    return re.sub(r"\s+", "", s)


def split_string_with_letter_prefix(s):
    # Initialize the index for where non-letter characters start
    index = 0
    # Loop through each character in the string
    for char in s:
        # Check if the character is a letter
        if char.isalpha():
            index += 1
        else:
            # Stop the loop once a non-letter is found
            break
    # Split the string into letters and the rest
    letters = s[:index]
    rest = s[index:]
    return letters, rest


def extract_prefix_map(tokens: Union[List[str], str]) -> OrderedDict[str, str]:
    if isinstance(tokens, str):
        tokens = tokens.split(" ")
    results: Dict[str, str] = OrderedDict()
    for token in tokens:
        if not token:
            continue
        key, value = split_string_with_letter_prefix(token)
        results[key] = value
    return results


def parse_hugin_control_points(lines: List[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    points0: List[Tuple[float, float]] = []
    points1: List[Tuple[float, float]] = []
    for line in lines:
        if line.startswith("c "):
            tokens = line.split(" ")
            tokens = [strip(t) for t in tokens]
            control_point = extract_prefix_map(tokens)
            pt0 = torch.tensor(
                [float(control_point["x"]), float(control_point["y"])], dtype=torch.float
            )
            pt1 = torch.tensor(
                [float(control_point["X"]), float(control_point["Y"])], dtype=torch.float
            )
            points0.append(pt0)
            points1.append(pt1)
    if points0:
        assert points1
        return torch.stack(points0), torch.stack(points1)
    return None


def configure_control_points(
    project_file_path: str,
    image0: str,
    image1: str,
    force: bool = False,
    output_directory: Optional[str] = None,
    use_hugin: bool = False,
) -> None:
    #  c n0 N1 x5162 y1173 X1416.1875 Y1252.78125 t0
    pto_file = load_pto_file(project_file_path)
    hugin_ctrl_points = parse_hugin_control_points(pto_file)
    if hugin_ctrl_points is None:
        use_hugin = False
    if hugin_ctrl_points is not None and not force:
        return

    control_points = None

    if use_hugin and hugin_ctrl_points is not None:
        m_kpts0 = hugin_ctrl_points[0]
        m_kpts1 = hugin_ctrl_points[1]
        control_points = dict(m_kpts0=m_kpts0, m_kpts1=m_kpts1)

    if not control_points:
        start = time.time()
        control_points = calculate_control_points(
            output_directory=output_directory, image0=image0, image1=image1
        )
        print(f"Calculated control points in {time.time() - start} seconds")

    if use_hugin and hugin_ctrl_points is not None:
        # Don't rewrite if we got tyhem from the huugin project file
        return

    pts0 = control_points["m_kpts0"]
    pts1 = control_points["m_kpts1"]
    assert len(pts0) == len(pts1)
    print(f"Found {len(pts0)} control points")
    assert len(pts0) and len(pts1)
    pto_file, _ = remove_control_points(lines=pto_file)
    pto_file.append("")
    pto_file.append(_CONTROL_POINTS_LINE)

    def _to_hugin_decimal(val: str) -> str:
        val = float(val)
        if val == float(int(val)):
            return f"{int(val)}"
        return f"{val:.12f}"

    for i in range(len(pts0)):
        point0 = [float(c) for c in pts0[i]]
        point1 = [float(c) for c in pts1[i]]
        line = f"c n0 N1 x{_to_hugin_decimal(point0[0])} y{_to_hugin_decimal(point0[1])} X{_to_hugin_decimal(point1[0])} Y{_to_hugin_decimal(point1[1])} t0"
        pto_file.append(line)
    save_pto_file(file_path=project_file_path, data=pto_file)
    configure_pano_size(
        project_file_path=project_file_path,
        pano_width=int(min(8192, image_width(cv2.imread(image0)) * 1.5)),
    )
    print("Done with control points")


def configure_pano_size(project_file_path: str, pano_width: int):
    if not os.path.exists(project_file_path):
        return
    pto_file = load_pto_file(project_file_path)
    # find line that starts with "p "
    index = None
    for i, line in enumerate(pto_file):
        if line.startswith("p "):
            index = i
            break
    if index is None:
        print("Could not find output pano properties line in pto file")
        return
    params = extract_prefix_map(pto_file[index])
    print(params)
    if int(params["w"]) == pano_width:
        return
    ar = float(params["w"]) / float(params["h"])
    w = pano_width
    h = int(w / ar)
    params["w"] = str(int(w))
    params["h"] = str(h)
    params["v"] = "180"
    output_line = ""
    for k, v in params.items():
        if output_line:
            output_line += " "
        output_line += k + v
    pto_file[index] = output_line
    save_pto_file(file_path=project_file_path, data=pto_file)
