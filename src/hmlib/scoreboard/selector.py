import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sys
from typing import List, Tuple


class ScoreboardSelector:
    def __init__(self, image_file: str) -> None:
        try:
            self.image: Image.Image = Image.open(image_file)
        except Exception as e:
            print(f"Error opening image file: {e}")
            sys.exit(1)

        self.root: tk.Tk = tk.Tk()
        self.root.title("Select Scoreboard Corners")

        self.canvas_width: int
        self.canvas_height: int
        self.canvas_width, self.canvas_height = self.image.size
        self.canvas: tk.Canvas = tk.Canvas(
            self.root, width=self.canvas_width, height=self.canvas_height
        )
        self.canvas.pack()

        self.tk_image: ImageTk.PhotoImage = ImageTk.PhotoImage(self.image)
        self.canvas_image: int = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.points: List[Tuple[int, int]] = []
        self.point_markers: List[int] = []
        self.lines: List[int] = []

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Delete>", self.on_key_press)

        button_frame: tk.Frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ok_button: tk.Button = tk.Button(button_frame, text="OK", command=self.process_ok)
        delete_button: tk.Button = tk.Button(
            button_frame, text="Delete", command=self.reset_selection
        )
        none_button: tk.Button = tk.Button(button_frame, text="None", command=self.root.quit)

        ok_button.pack(side=tk.LEFT)
        delete_button.pack(side=tk.LEFT)
        none_button.pack(side=tk.LEFT)

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
                line: int = self.canvas.create_line(
                    self.points[-2][0], self.points[-2][1], x, y, fill="red", width=2
                )
                self.lines.append(line)
            # If there are four points, draw a line from last to first to close the shape
            if len(self.points) == 4:
                line = self.canvas.create_line(
                    self.points[0][0], self.points[0][1], x, y, fill="red", width=2
                )
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
        bottom_two_sorted: List[Tuple[int, int]] = sorted(
            bottom_two, key=lambda p: p[0], reverse=True
        )
        # Now assign the points
        tl: Tuple[int, int] = top_two_sorted[0]
        tr: Tuple[int, int] = top_two_sorted[1]
        br: Tuple[int, int] = bottom_two_sorted[0]
        bl: Tuple[int, int] = bottom_two_sorted[1]
        return [tl, tr, br, bl]

    def process_ok(self) -> None:
        if len(self.points) != 4:
            messagebox.showinfo("Info", "Please select exactly 4 points.")
            return
        ordered_points: List[Tuple[int, int]] = self.order_points(self.points)
        # Print the points
        print("Selected points in clockwise order starting from the upper-left point:")
        for p in ordered_points:
            print(f"({p[0]}, {p[1]})")
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py imagefile")
    #     sys.exit(1)
    # image_file = sys.argv[1]
    image_file = "/mnt/ripper-data/Videos/ev-blackstars-ps/s.png"
    selector = ScoreboardSelector(image_file)
    selector.run()
