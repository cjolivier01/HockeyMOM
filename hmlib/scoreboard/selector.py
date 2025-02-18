import argparse
import os
import sys
import tkinter as tk
import traceback
from tkinter import messagebox
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from hmlib.config import (
    get_game_config,
    get_game_dir,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.utils.image import make_visible_image
from PIL import Image, ImageTk


class ScoreboardSelector:

    NULL_POINTS = [(0, 0), (0, 0), (0, 0), (0, 0)]

    def __init__(
        self,
        image: Union[Image.Image, np.ndarray],
        initial_points: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        try:
            if isinstance(image, Image.Image):
                self.image = image
            else:
                if isinstance(image, torch.Tensor):
                    if image.ndim == 4:
                        image = image[0]
                    # Make a CPU tensor since this sometimes causes CUDA OOM
                    # on low-memory GPU cards
                    image = make_visible_image(image.cpu())
                self.image: Image.Image = Image.fromarray(image)
        except Exception as e:
            traceback.print_exc()
            raise

        self.root: tk.Tk = tk.Tk()
        self.root.title("Select Scoreboard Corners")

        self.canvas_width: int
        self.canvas_height: int
        self.canvas_width, self.canvas_height = self.image.size
        self.canvas: tk.Canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.tk_image: ImageTk.PhotoImage = ImageTk.PhotoImage(self.image)
        self.canvas_image: int = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.points: List[Tuple[int, int]] = []
        self.point_markers: List[int] = []
        self.lines: List[int] = []

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Delete>", self.on_key_press)
        self.root.bind("<Key>", self.on_key_press)

        button_frame: tk.Frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Define a larger font for the buttons
        button_font = ("Helvetica", 16, "bold")

        ok_button: tk.Button = tk.Button(
            button_frame, text="OK", command=self.process_ok, font=button_font, width=10, height=2
        )
        delete_button: tk.Button = tk.Button(
            button_frame,
            text="Delete",
            command=self.reset_selection,
            font=button_font,
            width=10,
            height=2,
        )
        none_button: tk.Button = tk.Button(
            button_frame, text="None", command=self.root.quit, font=button_font, width=10, height=2
        )

        # Add padding to make the buttons larger and spaced out
        ok_button.pack(side=tk.LEFT, padx=10, pady=10)
        delete_button.pack(side=tk.LEFT, padx=10, pady=10)
        none_button.pack(side=tk.LEFT, padx=10, pady=10)

        if initial_points == ScoreboardSelector.NULL_POINTS:
            initial_points = []

        if initial_points and len(initial_points) == 4:
            self.points = initial_points
            self.draw_points_and_lines()
        elif initial_points:
            messagebox.showwarning("Warning", "Initial points provided are not exactly 4 points. Ignoring them.")

    def draw_points_and_lines(self) -> None:
        # Draw existing points and lines
        for i, (x, y) in enumerate(self.points):
            # Draw a small circle at the point
            r: int = 5
            point_marker: int = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
            self.point_markers.append(point_marker)
            # Draw line to the previous point
            if i > 0:
                line: int = self.canvas.create_line(
                    self.points[i - 1][0], self.points[i - 1][1], x, y, fill="red", width=2
                )
                self.lines.append(line)
        # Close the shape by connecting the last point to the first
        if len(self.points) == 4:
            line = self.canvas.create_line(
                self.points[-1][0],
                self.points[-1][1],
                self.points[0][0],
                self.points[0][1],
                fill="red",
                width=2,
            )
            self.lines.append(line)

    def on_click(self, event: tk.Event) -> None:
        if len(self.points) < 4:
            x: int = event.x
            y: int = event.y
            self.points.append((x, y))
            # Draw a small circle at the point
            r: int = 5
            point_marker: int = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
            self.point_markers.append(point_marker)
            # If there are at least two points, draw a line
            if len(self.points) > 1:
                line: int = self.canvas.create_line(self.points[-2][0], self.points[-2][1], x, y, fill="red", width=2)
                self.lines.append(line)
            # If there are four points, draw a line from last to first to close the shape
            if len(self.points) == 4:
                line = self.canvas.create_line(self.points[0][0], self.points[0][1], x, y, fill="red", width=2)
                self.lines.append(line)
        else:
            messagebox.showinfo("Info", "Already selected 4 points. Press Delete to reset.")

    def reset_selection(self) -> None:
        for marker in self.point_markers:
            self.canvas.delete(marker)
        for line in self.lines:
            self.canvas.delete(line)
        self.points = []
        self.point_markers = []
        self.lines = []

    def on_key_press(self, event: tk.Event) -> None:
        if event.keysym == "Delete":
            self.reset_selection()
        else:
            # Print out the ASCII code of the key pressed
            ascii_code = ord(event.char) if event.char else None
            if ascii_code is not None:
                if ascii_code == 27:
                    # ESC
                    self.process_none()
                elif ascii_code == 13:
                    # ENTER
                    self.process_ok()
                else:
                    print(f"Key pressed: {event.char}, ASCII code: {ascii_code}")

    def order_points(self, pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Sort the points based on their y-coordinate
        pts_sorted: List[Tuple[int, int]] = sorted(pts, key=lambda p: p[1])
        # The top two points
        top_two: List[Tuple[int, int]] = pts_sorted[:2]
        # The bottom two points
        bottom_two: List[Tuple[int, int]] = pts_sorted[2:]
        # Sort top two points based on x-coordinate
        top_two_sorted: List[Tuple[int, int]] = sorted(top_two, key=lambda p: p[0])
        # Sort bottom two in reverse order based on x-coordinate
        bottom_two_sorted: List[Tuple[int, int]] = sorted(bottom_two, key=lambda p: p[0], reverse=True)
        # Now assign the points
        tl: Tuple[int, int] = top_two_sorted[0]
        tr: Tuple[int, int] = bottom_two_sorted[1]
        br: Tuple[int, int] = bottom_two_sorted[0]
        bl: Tuple[int, int] = top_two_sorted[1]
        return [tl, tr, br, bl]

    def process_ok(self) -> None:
        if not self.points:
            # The non-selection set of points
            self.points = ScoreboardSelector.NULL_POINTS.copy()
        if len(self.points) != 4:
            messagebox.showinfo("Info", "Please select exactly 4 points.")
            return
        # ordered_points: List[Tuple[int, int]] = self.order_points(self.points)
        ordered_points: List[Tuple[int, int]] = self.points.copy()
        # Print the points
        print("Selected points in clockwise order starting from the upper-left point:")
        for p in ordered_points:
            print(f"({p[0]}, {p[1]})")
        self.points = ordered_points
        self.close()

    def process_none(self) -> None:
        self.points = ScoreboardSelector.NULL_POINTS.copy()
        self.close()

    def close(self):
        self.root.quit()
        self.root.destroy()  # Close the window completely

    def run(self) -> None:
        self.root.mainloop()


def parse_points(points_str_list: List[str]) -> Optional[List[Tuple[int, int]]]:
    if len(points_str_list) != 4:
        print("Error: Exactly four points must be provided.")
        return None
    points: List[Tuple[int, int]] = []
    for point_str in points_str_list:
        try:
            x_str, y_str = point_str.split(",")
            x = int(x_str)
            y = int(y_str)
            points.append((x, y))
        except ValueError:
            print(f"Error parsing point '{point_str}'. Points must be in the format x,y.")
            return None
    return points


def _untuple_points(points: List[Tuple[int, int]]) -> List[List[int]]:
    results: List[List[int, int]] = []
    for pt in points:
        assert len(pt) == 2
        results.append([pt[0], pt[1]])
    return results


def configure_scoreboard(game_id: str, image: Optional[torch.Tensor] = None, force: bool = False) -> List[List[int]]:
    assert game_id
    game_config = get_game_config(game_id=game_id)
    current_scoreboard = get_nested_value(game_config, "rink.scoreboard.perspective_polygon")
    if current_scoreboard and not force:
        return current_scoreboard

    if image is None:
        game_dir = get_game_dir(game_id=game_id)
        image_file = os.path.join(game_dir, "s.png")
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Could nto find image file: {image_file}")
        image = Image.open(image_file)
    selector = ScoreboardSelector(image=image, initial_points=current_scoreboard)
    selector.run()
    current_scoreboard = selector.points
    current_scoreboard = _untuple_points(current_scoreboard)
    set_nested_value(game_config, "rink.scoreboard.perspective_polygon", current_scoreboard)
    save_private_config(game_id=game_id, data=game_config, verbose=True)
    return current_scoreboard


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    configure_scoreboard(game_id="ev-blackstars-ps", force=True)
