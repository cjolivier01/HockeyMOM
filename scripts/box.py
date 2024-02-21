import os
import cv2
from pathlib import Path

from hmlib.config import get_game_config, save_game_config, set_nested_value
from hmlib.hm_opts import hm_opts

# import wx

# class ImageViewer(wx.Frame):
#     def __init__(self, parent, title):
#         super(ImageViewer, self).__init__(parent, title=title, size=(800, 600))
#         self.panel = wx.Panel(self)
#         self.points = []
#         self.Bind(wx.EVT_PAINT, self.on_paint)
#         self.panel.Bind(wx.EVT_LEFT_DOWN, self.on_click)

#         self.image_path = '/home/colivier/Videos/tvbb/panorama.tif'  # Replace with your image path
#         self.image = wx.Image(self.image_path, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
#         self.SetClientSize(self.image.GetSize())

#         self.Show()

#     def on_paint(self, event):
#         dc = wx.PaintDC(self.panel)
#         dc.DrawBitmap(self.image, 0, 0, True)
#         if len(self.points) > 1:
#             dc.SetPen(wx.Pen(wx.RED, 3))
#             for i in range(len(self.points) - 1):
#                 dc.DrawLine(self.points[i][0], self.points[i][1], self.points[i + 1][0], self.points[i + 1][1])

#     def on_click(self, event):
#         if len(self.points) < 4:
#             pos = event.GetPosition()
#             self.points.append([pos.x, pos.y])
#             self.Refresh()  # Trigger the paint event
#         if len(self.points) == 4:
#             self.Close()  # Close the frame
#             print(self.points)


def select_opencv(game_id: str):
    # Initialize global variables
    points = []  # List to store points

    def click_event(event, x, y, flags, params):
        # If left mouse button clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:  # Check if less than 4 points have been clicked
                points.append([x, y])  # Append the point
                cv2.circle(
                    img, (x, y), 3, (0, 255, 0), -1
                )  # Draw circle at the clicked point
                if len(points) > 1:
                    cv2.line(
                        img, tuple(points[-2]), tuple(points[-1]), (255, 0, 0), 2
                    )  # Draw line between points
                cv2.imshow("image", img)  # Show the image
            # if len(points) == 4:  # If 4 points have been clicked
            #     cv2.destroyAllWindows()  # Close the window

    # Load an image
    img = cv2.imread(
        # f'{os.environ["HOME"]}/Videos/{game_id}/panorama.tif'
        f'{os.environ["HOME"]}/Videos/{game_id}/s.png'
    )  # Replace 'path_to_your_image.jpg' with your image path
    cv2.imshow("image", img)

    # Set mouse callback function for window
    cv2.setMouseCallback("image", click_event)

    try:
        cv2.waitKey(0)  # Wait indefinitely for a key press
    except Exception as ex:
        print(ex)
    cv2.destroyAllWindows()  # Destroy all windows

    # Print the four points
    if len(points) == 4:
        print(points)
        return points
    return None


if __name__ == "__main__":
    # app = wx.App(False)
    # frame = ImageViewer(None, "Image Viewer")
    # app.MainLoop()
    opts = hm_opts()
    args = opts.parse()
    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / "..")
    game_config = get_game_config(game_id=args.game_id, root_dir=root_dir)

    points = select_opencv(args.game_id)

    if points and len(points) == 4:
        set_nested_value(game_config, "rink.scoreboard.perspective_polygon", points)
        save_game_config(game_id=args.game_id, root_dir=root_dir, data=game_config)
