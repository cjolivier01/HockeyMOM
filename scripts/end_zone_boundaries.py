import copy
import os
import tkinter as tk
from tkinter import Button, Label
from tkinter import font as tkfont
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from hmlib.camera.end_zones import load_lines_from_config
from hmlib.config import (
    get_game_config_private,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts


class ImageEditor:
    def __init__(self, game_id: str, image_path: str, lines: Dict[str, List[Tuple[int, int]]] = {}):
        self._game_id = game_id
        self.root = tk.Tk()
        self.root.title("End-Zone Threashold Drawer")

        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        # self.root.geometry(
        #     f"{self.photo.width()}x{self.photo.height() + 50}"
        # )  # 50 pixels additional height for controls

        self.canvas = tk.Canvas(self.root, width=self.photo.width(), height=self.photo.height())
        self.canvas.pack()

        self.label = Label(self.root, text="Choose an action and draw a line on the image.")
        self.label.pack()

        # Define a larger font for buttons
        self.button_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

        # Frame for buttons
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(side=tk.TOP, fill=tk.X)

        # Buttons
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(side=tk.TOP, fill=tk.X)
        self.add_button("Left Start", "left_start", "blue")
        self.add_button("Left Stop", "left_stop", "yellow")
        self.add_button("Right Start", "right_start", "blue")
        self.add_button("Right Stop", "right_stop", "yellow")
        self.done_button = Button(
            self.buttons_frame,
            text="Done",
            command=self.save_lines,
            font=self.button_font,
            height=2,
            width=10,
        )
        self.done_button.pack(side=tk.LEFT)

        # Line data
        self.lines: Dict[str, List[Tuple[int, int]]] = lines.copy()
        self.current_color: str = None
        self.current_label: str = None

        # Bind canvas clicks and key press
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<q>", self.quit)

        # Display the image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.start_point = None
        # Start with left_start
        self.set_draw_mode(label="left_start", color="green")
        self.set_current_title()
        self.redraw_lines()

    def set_current_title(self):
        self.root.title(f"End-Zone Threashold Drawer: Now selecting {self.current_label}...")

    def add_button(self, text, label, color):
        button = Button(
            self.buttons_frame,
            text=text,
            command=lambda: self.set_draw_mode(label, color),
            font=self.button_font,
            height=2,
            width=10,
        )
        # button = Button(
        #     self.buttons_frame, text=text, command=lambda: self.set_draw_mode(label, color)
        # )
        button.pack(side=tk.LEFT, padx=10, pady=10)

    def set_draw_mode(self, label, color):
        self.current_color = color
        self.current_label = label
        self.start_point = None
        self.set_current_title()

    def on_canvas_click(self, event):
        if self.start_point is None:
            self.start_point = (event.x, event.y)
        else:
            # Ensure the first point has the smaller x-value
            end_point = (event.x, event.y)
            sorted_points = sorted([self.start_point, end_point], key=lambda point: point[0])
            self.lines[self.current_label] = sorted_points
            self.redraw_lines()
            self.start_point = None

    def redraw_lines(self):
        self.canvas.delete("line")
        for label, points in self.lines.items():
            color = "green" if "start" in label else "yellow"
            self.canvas.create_line(*points[0], *points[1], fill=color, tags="line", width=2)

    def save_lines(self):
        for label, points in self.lines.items():
            print(f"{label}: {points}")
        if len(self.lines):
            game_config = get_game_config_private(game_id=args.game_id)
            for label, line in self.lines.items():
                if not line:
                    line = None
                set_nested_value(
                    game_config, f"rink.end_zones.{label}", [list(line[0]), list(line[1])]
                )
            save_private_config(game_id=self._game_id, data=game_config)

        self.quit()

    def quit(self, event=None):
        self.root.destroy()


def get_line(
    game_config: Dict[str, Any], key, dflt: List[Tuple[int, int]] = []
) -> List[Tuple[int, int]]:
    line = get_nested_value(game_config, key)
    if line is None:
        return copy.deepcopy(dflt)
    assert len(line) == 2
    return line


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    args.game_id = "ev-blackstars-ps"

    assert args.game_id

    game_config = get_game_config_private(game_id=args.game_id)

    lines: Dict[str, List[Tuple[int, int]]] = load_lines_from_config(config=game_config)

    image_path = f'{os.environ["HOME"]}/Videos/{args.game_id}/s.png'

    editor = ImageEditor(game_id=args.game_id, image_path=image_path, lines=lines)
    editor.root.mainloop()
