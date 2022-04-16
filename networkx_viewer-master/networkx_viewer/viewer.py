try:
    # Python 3
    import tkinter as tk
    import tkinter.messagebox as tkm
    import tkinter.simpledialog as tkd
except ImportError:
    # Python 2
    import Tkinter as tk
    import tkMessageBox as tkm
    import tkSimpleDialog as tkd

from networkx_viewer.graph_canvas import GraphCanvas
from networkx_viewer.tokens import TkPassthroughEdgeToken, TkPassthroughNodeToken

class ViewerApp(tk.Tk):
    def __init__(self, graph, **kwargs):
        tk.Tk.__init__(self)
        self.geometry('1000x600')
        self.title('NetworkX Viewer')

        bottom_row = 120
        self.columnconfigure(0, weight=1)
        self.rowconfigure(bottom_row, weight=1)

        self.canvas = GraphCanvas(graph, width=400, height=400, **kwargs)
        self.canvas.grid(row=0, column=0, rowspan=bottom_row+2, sticky='NESW')

class TkPassthroughViewerApp(ViewerApp):
    def __init__(self, graph, **kwargs):
        ViewerApp.__init__(self, graph,
            NodeTokenClass=TkPassthroughNodeToken,
            EdgeTokenClass=TkPassthroughEdgeToken, **kwargs)
