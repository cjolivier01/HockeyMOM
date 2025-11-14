import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python scripts/inspect_image.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]

try:
    # Load image
    img = Image.open(image_path)
    data = np.array(img)

    # Ensure we work with RGB for averaging; ignore alpha if present
    if data.ndim == 3 and data.shape[2] >= 3:
        rgb_data = data[:, :, :3]
    elif data.ndim == 2:  # grayscale -> replicate channel for consistent reporting
        rgb_data = np.stack([data, data, data], axis=-1)
    else:
        raise ValueError("Unsupported image format for averaging.")

    # Print whole-image average RGB at startup
    whole_mean = rgb_data.mean(axis=(0, 1))
    print(
        f"Whole image average RGB: R={whole_mean[0]:.2f}, G={whole_mean[1]:.2f}, B={whole_mean[2]:.2f}"
    )

    fig, ax = plt.subplots()
    ax.imshow(data)
    title_base = "Left-click pixel RGB; Shift+Left-drag to average region"
    ax.set_title(title_base)

    # On-screen legend
    legend_text = (
        "Controls:\n"
        "  • Left click: pixel RGB\n"
        "  • Shift+Left-drag: draw region; release prints average"
    )
    ax.text(
        0.01,
        0.99,
        legend_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="none"),
    )

    # State for right-button drag rectangle
    dragging = False
    start_x = start_y = None
    rect_patch = None

    height, width = rgb_data.shape[:2]

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # Left-click: display pixel values; Shift+Left: start rectangle drag
    def on_button_press(event):
        global dragging, start_x, start_y, rect_patch
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        def shift_held(ev):
            k = getattr(ev, "key", None)
            return isinstance(k, str) and ("shift" in k.lower())

        if event.button == 1 and shift_held(event):
            # Begin rectangle drag with Shift+Left
            dragging = True
            start_x, start_y = event.xdata, event.ydata
            if rect_patch is None:
                rect_patch = patches.Rectangle(
                    (start_x, start_y),
                    0,
                    0,
                    linewidth=2.0,
                    edgecolor="yellow",
                    facecolor="yellow",
                    alpha=0.25,
                    zorder=10,
                )
                ax.add_patch(rect_patch)
            else:
                rect_patch.set_xy((start_x, start_y))
                rect_patch.set_width(0)
                rect_patch.set_height(0)
            fig.canvas.draw()
            fig.canvas.flush_events()
        elif event.button == 1:
            # Simple left-click: print pixel
            x, y = int(event.xdata), int(event.ydata)
            x = clamp(x, 0, width - 1)
            y = clamp(y, 0, height - 1)
            pixel_value = data[y, x]
            if np.ndim(pixel_value) == 0:
                print(f"Pixel at ({x}, {y}): {pixel_value}")
            else:
                if pixel_value.shape[0] >= 3:
                    r, g, b = pixel_value[0], pixel_value[1], pixel_value[2]
                    print(f"Pixel at ({x}, {y}): R={r}, G={g}, B={b}")
                else:
                    print(f"Pixel at ({x}, {y}): {pixel_value}")

    def on_motion(event):
        global rect_patch
        if not dragging or rect_patch is None:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        cur_x, cur_y = event.xdata, event.ydata
        x0, y0 = start_x, start_y
        # Set rectangle by top-left and width/height
        x_min, y_min = min(x0, cur_x), min(y0, cur_y)
        width_px, height_px = abs(cur_x - x0), abs(cur_y - y0)
        rect_patch.set_xy((x_min, y_min))
        rect_patch.set_width(width_px)
        rect_patch.set_height(height_px)
        # While dragging, compute and display live average for current rectangle
        xi0 = int(np.floor(min(x0, cur_x)))
        yi0 = int(np.floor(min(y0, cur_y)))
        xi1 = int(np.ceil(max(x0, cur_x)))
        yi1 = int(np.ceil(max(y0, cur_y)))
        xi0 = clamp(xi0, 0, width - 1)
        yi0 = clamp(yi0, 0, height - 1)
        xi1 = clamp(xi1, 0, width - 1)
        yi1 = clamp(yi1, 0, height - 1)
        if xi1 < xi0:
            xi0, xi1 = xi1, xi0
        if yi1 < yi0:
            yi0, yi1 = yi1, yi0
        region = rgb_data[yi0 : yi1 + 1, xi0 : xi1 + 1]
        if region.size > 0:
            m = region.mean(axis=(0, 1))
            ax.set_title(
                f"{title_base} | Live avg RGB: R={m[0]:.2f}, G={m[1]:.2f}, B={m[2]:.2f}"
            )
        else:
            ax.set_title(title_base)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def on_button_release(event):
        global dragging, rect_patch
        if not dragging:
            return
        if event.button != 1:
            return
        dragging = False

        # Use the last mouse position if available; otherwise fall back to start
        end_x = event.xdata if event.xdata is not None else start_x
        end_y = event.ydata if event.ydata is not None else start_y
        if end_x is None or end_y is None or start_x is None or start_y is None:
            # Clean up any patch and return
            if rect_patch is not None:
                rect_patch.remove()
                rect_patch = None
                fig.canvas.draw_idle()
            return

        # Compute integer bounding box within image bounds
        x0 = int(np.floor(min(start_x, end_x)))
        y0 = int(np.floor(min(start_y, end_y)))
        x1 = int(np.ceil(max(start_x, end_x)))
        y1 = int(np.ceil(max(start_y, end_y)))

        x0 = clamp(x0, 0, width - 1)
        y0 = clamp(y0, 0, height - 1)
        x1 = clamp(x1, 0, width - 1)
        y1 = clamp(y1, 0, height - 1)

        # Ensure at least 1 pixel in each dimension; make x1/y1 inclusive for slicing
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        # Python slicing end-exclusive -> add +1
        region = rgb_data[y0 : y1 + 1, x0 : x1 + 1]
        if region.size > 0:
            mean_rgb = region.mean(axis=(0, 1))
            print(
                f"Region x=[{x0},{x1}], y=[{y0},{y1}] average RGB: "
                f"R={mean_rgb[0]:.2f}, G={mean_rgb[1]:.2f}, B={mean_rgb[2]:.2f}"
            )
        else:
            print("Empty selection; no pixels to average.")

        # Restore title and remove rectangle after release
        if rect_patch is not None:
            rect_patch.remove()
            rect_patch = None
            ax.set_title(title_base)
            fig.canvas.draw()
            fig.canvas.flush_events()

    # Connect events
    cid_press = fig.canvas.mpl_connect("button_press_event", on_button_press)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", on_motion)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_button_release)

    plt.show()

except FileNotFoundError:
    print(f"Error: The file at {image_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
