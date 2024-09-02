import subprocess

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
    # Name of the application window (This might need adjustment for exact match)
    app_name = 'Visual Studio Code'
    
    # Assuming two monitors each with 1920x1080 resolution side by side
    total_width = 1920 * 2
    total_height = 1080  # Adjust if your monitor heights are different
    
    window_ids = get_window_ids_by_name(app_name)
    if window_ids:
        resize_and_move_windows(window_ids, 0, 0, total_width, total_height)
    else:
        print("No windows found matching the name.")

if __name__ == "__main__":
    main()

