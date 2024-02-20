import wx

class ImageViewer(wx.Frame):
    def __init__(self, parent, title):
        super(ImageViewer, self).__init__(parent, title=title, size=(800, 600))
        self.panel = wx.Panel(self)
        self.points = []
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.on_click)

        self.image_path = '/home/colivier/Videos/tvbb/panorama.tif'  # Replace with your image path
        self.image = wx.Image(self.image_path, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.SetClientSize(self.image.GetSize())

        self.Show()

    def on_paint(self, event):
        dc = wx.PaintDC(self.panel)
        dc.DrawBitmap(self.image, 0, 0, True)
        if len(self.points) > 1:
            dc.SetPen(wx.Pen(wx.RED, 3))
            for i in range(len(self.points) - 1):
                dc.DrawLine(self.points[i][0], self.points[i][1], self.points[i + 1][0], self.points[i + 1][1])

    def on_click(self, event):
        if len(self.points) < 4:
            pos = event.GetPosition()
            self.points.append([pos.x, pos.y])
            self.Refresh()  # Trigger the paint event
        if len(self.points) == 4:
            self.Close()  # Close the frame
            print(self.points)

if __name__ == "__main__":
    app = wx.App(False)
    frame = ImageViewer(None, "Image Viewer")
    app.MainLoop()
