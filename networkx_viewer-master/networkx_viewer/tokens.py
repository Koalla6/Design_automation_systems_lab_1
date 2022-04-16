try:
    # Python 3
    import tkinter as tk
except ImportError:
    # Python 2
    import Tkinter as tk

class NodeToken(tk.Canvas):
    def __init__(self, host_canvas, data, node_name):
        tk.Canvas.__init__(self, width=20, height=20, highlightthickness=0)

        self._host_canvas = host_canvas
        self._complete = True
        self._marked = False
        self._default_bg = None
        self.render(data, node_name)

    def render(self, data, node_name):
        self.create_oval(5,5,15,15, fill='red',outline='black')

    def __getstate__(self):
        ans = {
            '_complete': self._complete,
            '_default_bg': self._default_bg,
            '_marked': self._marked,
        }
        return ans

class EdgeToken(object):
    def __init__(self, edge_data):
        self.edge_data = edge_data
        self._marked = False
        self._spline_id = None
        self._host_canvas = None

    def render(self, host_canvas, coords, cfg=None, directed=False):
        if cfg is None:
            cfg = self.render_cfg()
        # Amend config options to include options which must be included
        cfg['tags'] = 'edge'
        cfg['smooth'] = True
        if directed:
            # Add arrow
            cfg['arrow'] = tk.LAST
            cfg['arrowshape'] = (30,40,5)
        self._spline_id = host_canvas.create_line(*coords, **cfg)
        self._host_canvas = host_canvas

    def coords(self, coords):
        assert self._host_canvas is not None, "Must draw using render method first"
        return self._host_canvas.coords(self._spline_id, coords)

    def __getstate__(self):
        ans = {
            '_marked': self._marked,
        }
        return ans

class TkPassthroughNodeToken(NodeToken):
    def __init__(self, *args, **kwargs):
        self._default_label_color = 'black'
        self._default_outline_color = 'black'

        NodeToken.__init__(self, *args, **kwargs)


    def render(self, data, node_name):
        # Take a first cut at creating the marker and label
        self.label = self.create_text(0, 0, text=node_name)
        self.marker = self.create_oval(0, 0, 10, 10,
                                       fill='red',outline='black')

        # Modify marker using options from data
        cfg = self.itemconfig(self.marker)
        for k,v in cfg.copy().items():
            cfg[k] = data.get(k, cfg[k][-1])
        self.itemconfig(self.marker, **cfg)
        self._default_outline_color = data.get('outline',self._default_outline_color)

        # Modify the text label using options from data
        cfg = self.itemconfig(self.label)
        for k,v in cfg.copy().items():
            cfg[k] = data.get('label_'+k, cfg[k][-1])
        self.itemconfig(self.label, **cfg)
        self._default_label_color = data.get('label_fill',self._default_label_color)

        # Figure out how big we really need to be
        bbox = self.bbox(self.label)
        bbox = [abs(x) for x in bbox]
        br = ( max((bbox[0] + bbox[2]),20), max((bbox[1]+bbox[3]),20) )

        self.config(width=br[0], height=br[1]+7)

        # Place label and marker
        mid = ( int(br[0]/2.0), int(br[1]/2.0)+7 )
        self.coords(self.label, mid)
        self.coords(self.marker, mid[0]-5,0, mid[0]+5,10)


    def mark_complete(self):
        self._complete = True
        self.itemconfig(self.marker, outline=self._default_outline_color)
        self.itemconfig(self.label, fill=self._default_label_color)

class TkPassthroughEdgeToken(EdgeToken):
    _tk_line_options = [
        'stipple', 'activefill', 'joinstyle', 'dash',
        'disabledwidth', 'dashoffset', 'activewidth', 'fill', 'splinesteps',
        'offset', 'disabledfill', 'disableddash', 'width', 'state',
        'disabledstipple', 'activedash', 'tags', 'activestipple',
        'capstyle', 'arrowshape', 'smooth', 'arrow'
    ]
    _marked_width = 4.0

    def render_cfg(self):
        cfg = {}
        for k in self._tk_line_options:
            v = self.edge_data.get(k, None)
            if v:
                cfg[k] = v
        self._native_width = cfg.get('width', 1.0)
        return cfg
