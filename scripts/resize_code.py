#!python
import subprocess

from screeninfo import get_monitors


def get_window_ids_by_name(name):
    # Use xdotool to find all window IDs matching a specific name
    try:
        output = subprocess.check_output(['xdotool', 'search', '--name', name])
        window_ids = output.decode().strip().split()
        return window_ids
    except subprocess.CalledProcessError:
        return []

def resize_and_move_windows(window_ids, x, y, width, height):
    for window_id in window_ids:
        # Resize and move the window
        subprocess.call(['xdotool', 'windowmove', window_id, str(x), str(y)])
        subprocess.call(['xdotool', 'windowsize', window_id, str(width), str(height)])
        print(f'Resized and moved window ID {window_id}')

def main():
    # Name of the application window
    app_name = 'Visual Studio Code'

    # Get monitor information
    monitors = get_monitors()
    if len(monitors) < 2:
        print("Less than two monitors detected.")
        return

    # Calculate the total width and the maximum height of the left two monitors
    total_width = sum(monitor.width for monitor in monitors[:2])
    total_height = max(monitor.height for monitor in monitors[:2])

    window_ids = get_window_ids_by_name(app_name)
    if window_ids:
        resize_and_move_windows(window_ids, 0, 0, total_width, total_height)
    else:
        print("No windows found matching the name.")

if __name__ == "__main__":
    main()
