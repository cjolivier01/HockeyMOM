import os
import traceback
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from hmlib.config import (
    get_game_config,
    get_game_dir,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.utils.image import make_visible_image

# GUI backends
try:
    import tkinter as tk
    from tkinter import messagebox

    from PIL import ImageTk  # type: ignore

    _tk_available = True
except Exception:
    _tk_available = False

try:
    import cv2  # type: ignore

    _cv2_available = True
except Exception:
    _cv2_available = False


def get_max_screen_height() -> Optional[int]:
    if not _tk_available:
        return None
    root = None
    try:
        root = tk.Tk()  # type: ignore
        root.withdraw()  # type: ignore
        height = int(root.winfo_screenheight())  # type: ignore
        return height
    except Exception:
        return None
    finally:
        try:
            if root is not None:
                root.destroy()  # type: ignore
        except Exception:
            pass


class ScoreboardSelector:

    NULL_POINTS = [(0, 0), (0, 0), (0, 0), (0, 0)]

    def __init__(
        self,
        image: Union[Image.Image, np.ndarray],
        initial_points: Optional[List[Tuple[int, int]]] = None,
        max_display_height: Optional[int] = None,
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
        except Exception:
            traceback.print_exc()
            raise

        # Selection points state (shared across backends)
        self.points: List[Tuple[int, int]] = []

        self._original_size = self.image.size
        self._display_scale = 1.0
        if max_display_height is None:
            max_display_height = get_max_screen_height()
        if max_display_height is not None:
            max_h = max(max_display_height - 160, 1)
            scale = min(max_h / self._original_size[1], 1.0)
            if scale < 1.0:
                new_size = (
                    int(round(self._original_size[0] * scale)),
                    int(round(self._original_size[1] * scale)),
                )
                self.image = self.image.resize(new_size, Image.LANCZOS)
                self._display_scale = scale

        # Decide backend: prefer OpenCV (supports zoom/pan), fallback to Tk.
        self._backend: str = "none"
        self._tk_state = {}
        self._cv2_state = {}

        # Normalize initial points
        if initial_points == ScoreboardSelector.NULL_POINTS:
            initial_points = []
        if initial_points:
            initial_points = self._scale_points_for_display(initial_points)

        # Try to initialize OpenCV backend first.
        if _cv2_available:
            try:
                # Prepare image for OpenCV (BGR)
                img_np = np.array(self.image)
                if img_np.ndim == 2:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                self._cv2_state = dict(
                    img_bgr_orig=img_bgr,
                    window_name="Select Scoreboard Corners",
                )
                self._backend = "cv2"
                if initial_points and len(initial_points) == 4:
                    self.points = initial_points
            except Exception:
                traceback.print_exc()
                self._backend = "none"

        # Fallback to Tk backend when OpenCV is unavailable/unusable.
        if self._backend != "cv2" and _tk_available:
            root = None
            try:
                root = tk.Tk()  # type: ignore
                root.title("Select Scoreboard Corners")
                canvas_w, canvas_h = self.image.size
                canvas: tk.Canvas = tk.Canvas(root, width=canvas_w, height=canvas_h)  # type: ignore
                canvas.pack()

                tk_image = ImageTk.PhotoImage(self.image)  # type: ignore
                canvas_image = canvas.create_image(0, 0, anchor="nw", image=tk_image)

                # Bindings and buttons
                canvas.bind("<Button-1>", self.on_click)  # type: ignore
                root.bind("<Delete>", self.on_key_press)  # type: ignore
                root.bind("<Key>", self.on_key_press)  # type: ignore

                button_frame: tk.Frame = tk.Frame(root)  # type: ignore
                button_frame.pack(side=tk.BOTTOM, fill=tk.X)
                button_font = ("Helvetica", 16, "bold")

                ok_button: tk.Button = tk.Button(  # type: ignore
                    button_frame,
                    text="OK",
                    command=self.process_ok,
                    font=button_font,
                    width=10,
                    height=2,
                )
                delete_button: tk.Button = tk.Button(  # type: ignore
                    button_frame,
                    text="Delete",
                    command=self.reset_selection,
                    font=button_font,
                    width=10,
                    height=2,
                )
                none_button: tk.Button = tk.Button(  # type: ignore
                    button_frame,
                    text="None",
                    command=root.quit,
                    font=button_font,
                    width=10,
                    height=2,
                )
                ok_button.pack(side=tk.LEFT, padx=10, pady=10)
                delete_button.pack(side=tk.LEFT, padx=10, pady=10)
                none_button.pack(side=tk.LEFT, padx=10, pady=10)

                # Save Tk-specific state
                self._backend = "tk"
                self._tk_state = dict(
                    root=root,
                    canvas=canvas,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    tk_image=tk_image,
                    canvas_image=canvas_image,
                    point_markers=[],
                    lines=[],
                )

                # Draw initial points if any
                if initial_points and len(initial_points) == 4:
                    self.points = initial_points
                    self.draw_points_and_lines()
                elif initial_points:
                    messagebox.showwarning(  # type: ignore
                        "Warning",
                        "Initial points provided are not exactly 4 points. Ignoring them.",
                    )

            except Exception:
                try:
                    if root is not None:
                        root.destroy()  # type: ignore
                except Exception:
                    pass
                self._backend = "none"

        if self._backend == "none":
            raise RuntimeError(
                "No available GUI backend to select scoreboard points (Tkinter unusable and OpenCV not available)."
            )

    def _scale_points_for_display(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not points:
            return []
        if self._display_scale == 1.0:
            return [(int(pt[0]), int(pt[1])) for pt in points]
        scale = self._display_scale
        return [(int(round(pt[0] * scale)), int(round(pt[1] * scale))) for pt in points]

    def _scale_points_to_original(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not points:
            return []
        if self._display_scale == 1.0:
            return [(int(pt[0]), int(pt[1])) for pt in points]
        inv_scale = 1.0 / self._display_scale
        return [(int(round(pt[0] * inv_scale)), int(round(pt[1] * inv_scale))) for pt in points]

    def draw_points_and_lines(self) -> None:
        if self._backend == "tk":
            canvas: tk.Canvas = self._tk_state["canvas"]  # type: ignore
            point_markers: List[int] = self._tk_state["point_markers"]  # type: ignore
            lines: List[int] = self._tk_state["lines"]  # type: ignore
            for i, (x, y) in enumerate(self.points):
                r: int = 5
                point_marker: int = canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
                point_markers.append(point_marker)
                if i > 0:
                    line: int = canvas.create_line(
                        self.points[i - 1][0], self.points[i - 1][1], x, y, fill="red", width=2
                    )
                    lines.append(line)
            if len(self.points) == 4:
                line = canvas.create_line(
                    self.points[-1][0],
                    self.points[-1][1],
                    self.points[0][0],
                    self.points[0][1],
                    fill="red",
                    width=2,
                )
                lines.append(line)
        elif self._backend == "cv2":
            # Drawing handled during refresh loop
            pass

    def on_click(self, event) -> None:
        if self._backend == "tk":
            if len(self.points) < 4:
                x: int = event.x
                y: int = event.y
                self.points.append((x, y))
                r: int = 5
                canvas: tk.Canvas = self._tk_state["canvas"]  # type: ignore
                point_markers: List[int] = self._tk_state["point_markers"]  # type: ignore
                lines: List[int] = self._tk_state["lines"]  # type: ignore
                point_marker: int = canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
                point_markers.append(point_marker)
                if len(self.points) > 1:
                    line: int = canvas.create_line(
                        self.points[-2][0], self.points[-2][1], x, y, fill="red", width=2
                    )
                    lines.append(line)
                if len(self.points) == 4:
                    line = canvas.create_line(
                        self.points[0][0], self.points[0][1], x, y, fill="red", width=2
                    )
                    lines.append(line)
            else:
                try:
                    messagebox.showinfo("Info", "Already selected 4 points. Press Delete to reset.")  # type: ignore
                except Exception:
                    print("Already selected 4 points. Press Delete to reset.")
        # For cv2 backend, clicks are handled via setMouseCallback within run()

    def reset_selection(self) -> None:
        if self._backend == "tk":
            canvas: tk.Canvas = self._tk_state["canvas"]  # type: ignore
            for marker in list(self._tk_state["point_markers"]):  # type: ignore
                canvas.delete(marker)
            for line in list(self._tk_state["lines"]):  # type: ignore
                canvas.delete(line)
            self._tk_state["point_markers"] = []  # type: ignore
            self._tk_state["lines"] = []  # type: ignore
        self.points = []

    def on_key_press(self, event) -> None:
        if self._backend == "tk":
            if event.keysym == "Delete":
                self.reset_selection()
            else:
                ascii_code = ord(event.char) if getattr(event, "char", None) else None
                if ascii_code is not None:
                    if ascii_code == 27:
                        self.process_none()
                    elif ascii_code == 13:
                        self.process_ok()
                    else:
                        print(f"Key pressed: {event.char}, ASCII code: {ascii_code}")

    def order_points_clockwise(self, pts: torch.Tensor):
        # Ensure pts is a NumPy array of shape (4, 2)
        if isinstance(pts, torch.Tensor):
            pts = pts.to(torch.float32).cpu().numpy()
        elif isinstance(pts, list):
            pts = np.array(pts, dtype=np.float32)
        else:
            pts = pts.astype(np.float32)

        # Compute the sum and difference of the points.
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Allocate an array for the ordered points: [top-left, top-right, bottom-right, bottom-left]
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]  # top-left: smallest sum
        ordered[2] = pts[np.argmax(s)]  # bottom-right: largest sum
        ordered[1] = pts[np.argmin(diff)]  # top-right: smallest difference
        ordered[3] = pts[np.argmax(diff)]  # bottom-left: largest difference

        return list(map(list, ordered))

    def process_ok(self) -> None:
        if not self.points:
            # The non-selection set of points
            self.points = ScoreboardSelector.NULL_POINTS.copy()
        if len(self.points) != 4:
            try:
                if self._backend == "tk":
                    messagebox.showinfo("Info", "Please select exactly 4 points.")  # type: ignore
            except Exception:
                pass
            return
        ordered_points: List[Tuple[int, int]] = self.order_points_clockwise(self.points)
        ordered_points = self._scale_points_to_original(ordered_points)
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
        if self._backend == "tk":
            try:
                root: tk.Tk = self._tk_state.get("root")  # type: ignore
                if root is not None:
                    root.quit()  # type: ignore
                    root.destroy()  # type: ignore
            except Exception:
                pass
        elif self._backend == "cv2":
            try:
                cv2.destroyWindow(self._cv2_state.get("window_name", "Select Scoreboard Corners"))  # type: ignore
            except Exception:
                pass

    def run(self) -> None:
        if self._backend == "tk":
            root: tk.Tk = self._tk_state["root"]  # type: ignore
            root.mainloop()
        elif self._backend == "cv2":
            self._run_cv2()

    def _run_cv2(self) -> None:
        assert self._backend == "cv2"

        win = self._cv2_state["window_name"]
        img_bgr_orig = self._cv2_state["img_bgr_orig"].copy()
        img_h, img_w = img_bgr_orig.shape[:2]

        zoom: float = 1.0
        min_zoom: float = 1.0
        max_zoom: float = 20.0
        offset_x: float = 0.0
        offset_y: float = 0.0
        pan_active: bool = False
        pan_start: Tuple[int, int] = (0, 0)
        pan_origin: Tuple[float, float] = (0.0, 0.0)

        view_w: int = img_w
        view_h: int = img_h
        crop_w: int = img_w
        crop_h: int = img_h

        def _safe_window_size() -> Tuple[int, int]:
            try:
                _, _, w, h = cv2.getWindowImageRect(win)
                if w > 1 and h > 1:
                    return int(w), int(h)
            except Exception:
                pass
            return img_w, img_h

        def _update_view_geometry() -> None:
            nonlocal view_w, view_h, crop_w, crop_h
            view_w, view_h = _safe_window_size()
            crop_w = min(img_w, max(1, int(round(float(view_w) / max(zoom, 1e-6)))))
            crop_h = min(img_h, max(1, int(round(float(view_h) / max(zoom, 1e-6)))))

        def _clamp_offsets() -> None:
            nonlocal offset_x, offset_y
            max_x = max(0.0, float(img_w - crop_w))
            max_y = max(0.0, float(img_h - crop_h))
            offset_x = min(max(offset_x, 0.0), max_x)
            offset_y = min(max(offset_y, 0.0), max_y)

        def _screen_to_image(x: int, y: int) -> Tuple[float, float]:
            sx = float(x)
            sy = float(y)
            ix = offset_x + sx * (float(crop_w) / max(float(view_w), 1.0))
            iy = offset_y + sy * (float(crop_h) / max(float(view_h), 1.0))
            ix = min(max(ix, 0.0), float(max(img_w - 1, 0)))
            iy = min(max(iy, 0.0), float(max(img_h - 1, 0)))
            return ix, iy

        # Mouse callback
        def on_mouse(event, x, y, flags, param):
            nonlocal zoom, offset_x, offset_y, pan_active, pan_start, pan_origin
            _update_view_geometry()
            _clamp_offsets()
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.points) < 4:
                    px, py = _screen_to_image(x, y)
                    self.points.append((int(round(px)), int(round(py))))
                else:
                    print("Already selected 4 points. Press 'd' to reset.")
            elif event == cv2.EVENT_MBUTTONDOWN:
                pan_active = True
                pan_start = (int(x), int(y))
                pan_origin = (offset_x, offset_y)
            elif event == cv2.EVENT_MBUTTONUP:
                pan_active = False
            elif event == cv2.EVENT_MOUSEMOVE and pan_active:
                dx = float(x - pan_start[0]) * (float(crop_w) / max(float(view_w), 1.0))
                dy = float(y - pan_start[1]) * (float(crop_h) / max(float(view_h), 1.0))
                offset_x = pan_origin[0] - dx
                offset_y = pan_origin[1] - dy
                _clamp_offsets()
            elif event == cv2.EVENT_MOUSEWHEEL:
                wheel_delta = cv2.getMouseWheelDelta(flags)
                if wheel_delta == 0:
                    return
                anchor_x, anchor_y = _screen_to_image(x, y)
                zoom_step = 1.15
                wheel_steps = float(wheel_delta) / 120.0
                if wheel_steps > 0:
                    zoom = min(max_zoom, zoom * (zoom_step**wheel_steps))
                else:
                    zoom = max(min_zoom, zoom / (zoom_step ** abs(wheel_steps)))
                _update_view_geometry()
                offset_x = anchor_x - float(x) * (float(crop_w) / max(float(view_w), 1.0))
                offset_y = anchor_y - float(y) * (float(crop_h) / max(float(view_h), 1.0))
                _clamp_offsets()

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, img_w, img_h)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            _update_view_geometry()
            _clamp_offsets()

            x0 = int(round(offset_x))
            y0 = int(round(offset_y))
            x0 = min(max(x0, 0), max(img_w - crop_w, 0))
            y0 = min(max(y0, 0), max(img_h - crop_h, 0))
            # Keep transform math consistent with the effective crop origin.
            offset_x, offset_y = float(x0), float(y0)
            crop = img_bgr_orig[y0 : y0 + crop_h, x0 : x0 + crop_w]
            if crop.size == 0:
                crop = img_bgr_orig
                x0, y0, crop_h, crop_w = 0, 0, img_h, img_w
                offset_x, offset_y = 0.0, 0.0
            if crop.shape[1] != view_w or crop.shape[0] != view_h:
                img = cv2.resize(crop, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
            else:
                img = crop.copy()

            def _image_to_screen(pt: Tuple[int, int]) -> Tuple[int, int]:
                px, py = float(pt[0]), float(pt[1])
                sx = int(round((px - offset_x) * (float(view_w) / max(float(crop_w), 1.0))))
                sy = int(round((py - offset_y) * (float(view_h) / max(float(crop_h), 1.0))))
                return sx, sy

            # Draw points and lines in the current zoom/pan view.
            for i, pt in enumerate(self.points):
                sx, sy = _image_to_screen(pt)
                cv2.circle(img, (sx, sy), 5, (0, 0, 255), -1)
                if i > 0:
                    sx0, sy0 = _image_to_screen(self.points[i - 1])
                    cv2.line(img, (sx0, sy0), (sx, sy), (0, 0, 255), 2)
            if len(self.points) == 4:
                sx0, sy0 = _image_to_screen(self.points[0])
                sx1, sy1 = _image_to_screen(self.points[-1])
                cv2.line(img, (sx0, sy0), (sx1, sy1), (0, 0, 255), 2)

            # Instructions overlay
            instr0 = "LClick: add corner  Wheel: zoom  MWheel+drag: pan"
            instr1 = "ENTER: OK  d: reset points  r: reset view  ESC: None"
            cv2.putText(img, instr0, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, instr1, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            status = (
                f"zoom={zoom:.2f}x offset=({int(round(offset_x))}, {int(round(offset_y))}) "
                f"points={len(self.points)}/4"
            )
            cv2.putText(
                img,
                status,
                (10, max(86, int(view_h - 14))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(win, img)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                self.process_none()
                break
            if key in (10, 13):  # Enter
                self.process_ok()
                if len(self.points) == 4 or self.points == ScoreboardSelector.NULL_POINTS:
                    break
            if key in (ord("d"), ord("D")):
                self.reset_selection()
            if key in (ord("r"), ord("R")):
                zoom = 1.0
                offset_x = 0.0
                offset_y = 0.0

        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


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
        results.append([int(pt[0]), int(pt[1])])
    return results


def configure_scoreboard(
    game_id: str,
    image: Optional[torch.Tensor] = None,
    force: bool = False,
    max_display_height: Optional[int] = None,
) -> List[List[int]]:
    assert game_id
    game_config = get_game_config(game_id=game_id)
    current_scoreboard = get_nested_value(game_config, "rink.scoreboard.perspective_polygon")
    if current_scoreboard and not force:
        return current_scoreboard

    if image is None:
        game_dir = get_game_dir(game_id=game_id)
        image_file = os.path.join(game_dir, "s.png")
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Could not find image file: {image_file}")
        image = Image.open(image_file)
    selector = ScoreboardSelector(
        image=image,
        initial_points=current_scoreboard,
        max_display_height=max_display_height,
    )
    selector.run()
    current_scoreboard = selector.points
    current_scoreboard = _untuple_points(current_scoreboard)
    set_nested_value(game_config, "rink.scoreboard.perspective_polygon", current_scoreboard)
    save_private_config(game_id=game_id, data=game_config, verbose=True)
    return current_scoreboard


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    configure_scoreboard(game_id=args.game_id, force=True)
