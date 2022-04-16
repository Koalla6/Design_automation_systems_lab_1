from math import atan2, pi, cos, sin
#import collections
import pickle
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

import networkx as nx

from networkx_viewer.tokens import NodeToken, EdgeToken

from functools import wraps
def undoable(func):
    """Wrapper to create a savepoint which can be revered to using the
    GraphCanvas.undo method."""
    @wraps(func)
    def _wrapper(*args, **kwargs):
        # First argument should be the graphcanvas object (ie, "self")
        self = args[0]
        if not self._undo_suspend:
            self._undo_suspend = True # Prevent chained undos
            self._undo_states.append(self.dump_visualization())
            # Anytime we do an undoable action, the redo tree gets wiped
            self._redo_states = []
            func(*args, **kwargs)
            self._undo_suspend = False
        else:
            func(*args, **kwargs)
    return _wrapper

class GraphCanvas(tk.Canvas):
    """Expandable GUI to plot a NetworkX Graph"""

    def __init__(self, graph, **kwargs):
        """
        kwargs specific to GraphCanvas:
            - NodeTokenClass = Class to instantiate for a new node
               widget.  Should be inherited from NodeToken (which is from
               tk.Canvas)
            - EdgeTokenClass = Class to instantiate for a new edge widget.
               Should be inherited from EdgeToken.
            - home_node = Node to plot around when first rendering canvas
            - levels = How many nodes out to also plot when rendering

        """
        ###
        # Deal with the graph
        ###

        # Raw data graph
        self.dataG = graph

        # Graph representting what subsect of the data graph currently being
        #  displayed.
        self.dispG = nx.MultiGraph()

        # this data is used to keep track of an
        # item being dragged
        self._drag_data = {'x': 0, 'y': 0, 'item': None}

        # This data is used to track panning objects (x,y coords)
        self._pan_data = (None, None)

        # List of filters to run whenever trying to add a node to the graph
        self._node_filters = []

        # Undo list
        self._undo_states = []
        self._redo_states = []
        self._undo_suspend = False

        # Create a display version of this graph
        # If requested, plot only within a certain level of the home node
        home_node = kwargs.pop('home_node', None)
        if home_node:
            levels = kwargs.pop('levels', 1)
            graph = self._neighbors(home_node, levels=levels, graph=graph)

        # Class to use when create a node widget
        self._NodeTokenClass = kwargs.pop('NodeTokenClass',
                                          NodeToken)
        assert issubclass(self._NodeTokenClass, NodeToken), \
            "NodeTokenClass must be inherited from NodeToken"
        self._EdgeTokenClass = kwargs.pop('EdgeTokenClass',
                                          EdgeToken)
        assert issubclass(self._EdgeTokenClass, EdgeToken), \
            "NodeTokenClass must be inherited from NodeToken"

        ###
        # Now we can do UI things
        ###
        tk.Canvas.__init__(self, **kwargs)

        self._plot_graph(graph)

        # Center the plot on the home node or first node in graph
        self.center_on_node(home_node or next(iter(graph.nodes())))

        # add bindings for clicking, dragging and releasing over
        # any object with the "node" tammg
        # self.tag_bind('node', '<ButtonPress-1>', self.onNodeButtonPress)
        # self.tag_bind('node', '<ButtonRelease-1>', self.onNodeButtonRelease)
        # self.tag_bind('node', '<B1-Motion>', self.onNodeMotion)
        #
        # self.tag_bind('edge', '<Button-1>', self.onEdgeClick)
        # self.tag_bind('edge', '<Button-3>', self.onEdgeRightClick)
        #
        self.bind('<ButtonPress-1>', self.onPanStart)
        self.bind('<ButtonRelease-1>', self.onPanEnd)
        self.bind('<B1-Motion>', self.onPanMotion)
        #
        self.bind_all('<MouseWheel>', self.onZoom)

    def _draw_edge(self, u, v):
        frm_disp = self._find_disp_node(u)
        to_disp = self._find_disp_node(v)

        directed = False

        if isinstance(self.dataG, nx.MultiDiGraph):
            directed = True
            edges = self.dataG.get_edge_data(u, v)
        elif isinstance(self.dataG, nx.DiGraph):
            directed = True
            edges = {0: self.dataG.edges[u, v]}
        elif isinstance(self.dataG, nx.MultiGraph):
            edges = self.dataG.get_edge_data(u, v)
        elif isinstance(self.dataG, nx.Graph):
            edges = {0: self.dataG.edges[u, v]}
        else:
            raise NotImplementedError('Data Graph Type not Supported')

        # Figure out edge arc distance multiplier
        if len(edges) == 1:
            m = 0
        else:
            m = 15

        for key, data in edges.items():
            token = self._EdgeTokenClass(data)
            if isinstance(self.dataG, nx.MultiGraph):
                dataG_id = (u,v,key)
            elif isinstance(self.dataG, nx.Graph):
                dataG_id = (u,v)
            self.dispG.add_edge(frm_disp, to_disp, key, dataG_id=dataG_id, dispG_frm=frm_disp, token=token, m=m)

            x1,y1 = self._node_center(frm_disp)
            x2,y2 = self._node_center(to_disp)
            xa,ya = self._spline_center(x1,y1,x2,y2,m)

            token.render(host_canvas=self, coords=(x1,y1,xa,ya,x2,y2),
                         directed=directed)

            if m > 0:
                m = -m # Flip sides
            else:
                m = -(m+m)  # Go next increment out

    def _draw_node(self, coord, data_node):
        """Create a token for the data_node at the given coordinater"""
        (x,y) = coord
        data = self.dataG.nodes[data_node]

        # Apply filter to node to make sure we should draw it
        for filter_lambda in self._node_filters:
            try:
                draw_flag = eval(filter_lambda, {'u':data_node, 'd':data})
            except Exception as e:
                self._show_filter_error(filter_lambda, e)
                return
            # Filters are applied as an AND (ie, all must be true)
            # So if one is false, exit
            if draw_flag == False:
                return

        # Create token and draw node
        token = self._NodeTokenClass(self, data, data_node)
        id = self.create_window(x, y, window=token, anchor=tk.CENTER,
                                  tags='node')
        self.dispG.add_node(id, dataG_id=data_node,
                                 token_id=id, token=token)
        return id

    def _node_center(self, item_id):
        """Calcualte the center of a given node"""
        b = self.bbox(item_id)
        return ( (b[0]+b[2])/2, (b[1]+b[3])/2 )

    def _spline_center(self, x1, y1, x2, y2, m):
        """Given the coordinate for the end points of a spline, calcuate
        the mipdoint extruded out m pixles"""
        a = (x2 + x1)/2
        b = (y2 + y1)/2
        beta = (pi/2) - atan2((y2-y1), (x2-x1))

        xa = a - m*cos(beta)
        ya = b + m*sin(beta)
        return (xa, ya)

    @undoable
    def onPanStart(self, event):
        self._pan_data = (event.x, event.y)
        self.winfo_toplevel().config(cursor='fleur')

    def onPanMotion(self, event):
        # compute how much to move
        delta_x = event.x - self._pan_data[0]
        delta_y = event.y - self._pan_data[1]
        self.move(tk.ALL, delta_x, delta_y)

        # Record new location
        self._pan_data = (event.x, event.y)

    def onPanEnd(self, event):
        self._pan_data = (None, None)
        self.winfo_toplevel().config(cursor='arrow')

    def onZoom(self, event):
        factor = 0.1 * (1 if event.delta < 0 else -1)

        # Translate root coordinates into relative coordinates
        x = (event.widget.winfo_rootx() + event.x) - self.winfo_rootx()
        y = (event.widget.winfo_rooty() + event.y) - self.winfo_rooty()

        # Move everyone proportional to how far they are from the cursor
        ids = self.find_withtag('node') # + self.find_withtag('edge')

        for i in ids:
            ix, iy, t1, t2 = self.bbox(i)

            dx = (x-ix)*factor
            dy = (y-iy)*factor

            self.move(i, dx, dy)

        # Redraw all the edges
        for to_node, from_node, data in self.dispG.edges(data=True):
            from_xy = self._node_center(from_node)
            to_xy = self._node_center(to_node)
            if data['dispG_frm'] != from_node:
                # Flip!
                a = from_xy[:]
                from_xy = to_xy[:]
                to_xy = a[:]
            spline_xy = self._spline_center(*from_xy+to_xy+(data['m'],))

            data['token'].coords((from_xy+spline_xy+to_xy))

    @undoable
    def center_on_node(self, data_node):
        """Center canvas on given **DATA** node"""
        try:
            disp_node = self._find_disp_node(data_node)
        except ValueError as e:
            tkm.showerror("Unable to find node", str(e))
            return
        x,y = self.coords(self.dispG.nodes[disp_node]['token_id'])

        # Find center of canvas
        w = self.winfo_width()/2
        h = self.winfo_height()/2
        if w < 1:
            # We haven't been drawn yet
            w = int(self['width'])/2
            h = int(self['height'])/2

        # Calc delta to move to center
        delta_x = w - x
        delta_y = h - y

        self.move(tk.ALL, delta_x, delta_y)

    def dump_visualization(self):
        """Record currently visable nodes, their position, and their widget's
        state.  Used by undo functionality and to memorize speicific displays"""

        ans = self.dispG.copy()

        # Add current x,y info to the graph
        for n, d in ans.nodes(data=True):
            (d['x'],d['y']) = self.coords(d['token_id'])

        # Pickle the whole thing up
        ans = pickle.dumps(ans)

        return ans

    def _plot_graph(self, graph):
        # Create nodes
        scale = min(self.winfo_width(), self.winfo_height())
        if scale == 1:
            # Canvas not initilized yet; use height and width hints
            scale = int(min(self['width'], self['height']))

        scale -= 50
        if len(graph) > 1:
            layout = self.create_layout(graph, scale=scale, min_distance=50)

            # Find min distance between any node and make sure that is at least
            #  as big as
            for n in graph.nodes():
                self._draw_node(layout[n]+20, n)
        else:
            self._draw_node((scale/2, scale/2), list(graph.nodes())[0])

        # Create edges
        for frm, to in set(graph.edges()):
            self._draw_edge(frm, to)

        self._graph_changed()

    def _graph_changed(self):
        for n, d in self.dispG.nodes(data=True):
            token = d['token']
            if self.dispG.degree(n) == self.dataG.degree(d['dataG_id']):
                token.mark_complete()
            else:
                token.mark_incomplete()


    def _find_disp_node(self, data_node):
        disp_node = [a for a, d in self.dispG.nodes(data=True)
                    if d['dataG_id'] == data_node]
        if len(disp_node) == 0 and str(data_node).isdigit():
            # Try again, this time using the int version
            data_node = int(data_node)
            disp_node = [a for a, d in self.dispG.nodes(data=True)
                    if d['dataG_id'] == data_node]

        if len(disp_node) == 0:
            # It could be that this node is not displayed because it is
            #  currently being filtered out.  Test for that and, if true,
            #  raise a NodeFiltered exception
            for f in self._node_filters:
                try:
                    show_flag = eval(f, {'u':data_node,
                                         'd':self.dataG.nodes[data_node]})
                except Exception as e:
                    # Usually we we would alert user that eval failed, but
                    #  in this case, we're doing this without their knowlage
                    #  so we're just going to die silently
                    break
                if show_flag == False:
                    raise NodeFiltered
            raise ValueError("Data Node '%s' is not currently displayed"%\
                                data_node)
        elif len(disp_node) != 1:
            raise AssertionError("Data node '%s' is displayed multiple "
                                    "times" % data_node)
        return disp_node[0]

    def create_layout(self, G, pos=None, fixed=None, scale=1.0,
                      min_distance=None):
        dim = 2

        try:
            import numpy as np
        except ImportError:
            raise ImportError("fruchterman_reingold_layout() requires numpy: http://scipy.org/ ")
        if fixed is not None:
            nfixed=dict(zip(G,range(len(G))))
            fixed=np.asarray([nfixed[v] for v in fixed])

        if pos is not None:
            # Determine size of exisiting domain
            dom_size = max(flatten(pos.values()))
            pos_arr=np.asarray(np.random.random((len(G),dim)))*dom_size
            for i,n in enumerate(G):
                if n in pos:
                    pos_arr[i]=np.asarray(pos[n])
        else:
            pos_arr=None
            dom_size = 1.0

        if len(G)==0:
            return {}
        if len(G)==1:
            return {G.nodes()[0]:(1,)*dim}

        A=nx.to_numpy_matrix(G)
        nnodes,_ = A.shape
        # I've found you want to occupy about a two-thirds of the window size
        if fixed is not None:
            k=(min(self.winfo_width(), self.winfo_height())*.66)/np.sqrt(nnodes)
        else:
            k = None

        # Alternate k, for when vieweing the whole graph, not a subset
        #k=dom_size/np.sqrt(nnodes)
        pos=self._fruchterman_reingold(A,dim,k,pos_arr,fixed)

        if fixed is None:
            # Only rescale non fixed layouts
            pos= nx.layout.rescale_layout(pos,scale=scale)

        if min_distance and fixed is None:
            # Find min distance between any two nodes and scale such that
            #  this distance = min_distance

            # matrix of difference between points
            delta = np.zeros((pos.shape[0],pos.shape[0],pos.shape[1]),
                             dtype=A.dtype)
            for i in range(pos.shape[1]):
                delta[:,:,i]= pos[:,i,None]-pos[:,i]
            # distance between points
            distance=np.sqrt((delta**2).sum(axis=-1))

            cur_min_dist = np.where(distance==0, np.inf, distance).min()

            if cur_min_dist < min_distance:
                # calculate scaling factor and rescale
                rescale = (min_distance / cur_min_dist) * pos.max()

                pos = nx.layout.rescale_layout(pos, scale=rescale)

        return dict(zip(G,pos))

    def _fruchterman_reingold(self, A, dim=2, k=None, pos=None, fixed=None,
                              iterations=50):
        # Position nodes in adjacency matrix A using Fruchterman-Reingold
        # Entry point for NetworkX graph is fruchterman_reingold_layout()
        try:
            import numpy as np
        except ImportError:
            raise ImportError("_fruchterman_reingold() requires numpy: http://scipy.org/ ")

        try:
            nnodes,_=A.shape
        except AttributeError:
            raise nx.NetworkXError(
                "fruchterman_reingold() takes an adjacency matrix as input")

        A=np.asarray(A) # make sure we have an array instead of a matrix

        if pos is None:
            # random initial positions
            pos=np.asarray(np.random.random((nnodes,dim)),dtype=A.dtype)
        else:
            # make sure positions are of same type as matrix
            pos=pos.astype(A.dtype)

        # optimal distance between nodes
        if k is None:
            k=np.sqrt(1.0/nnodes)
        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        # Modified to actually detect for domain area
        t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt=t/float(iterations+1)
        delta = np.zeros((pos.shape[0],pos.shape[0],pos.shape[1]),dtype=A.dtype)
        # the inscrutable (but fast) version
        # this is still O(V^2)
        # could use multilevel methods to speed this up significantly
        for iteration in range(iterations):
            # matrix of difference between points
            for i in range(pos.shape[1]):
                delta[:,:,i]= pos[:,i,None]-pos[:,i]
            # distance between points
            distance=np.sqrt((delta**2).sum(axis=-1))
            # enforce minimum distance of 0.01
            distance=np.where(distance<0.01,0.01,distance)
            # displacement "force"
            displacement=np.transpose(np.transpose(delta)*\
                                      (k*k/distance**2-A*distance/k))\
                                      .sum(axis=1)
            # update positions
            length=np.sqrt((displacement**2).sum(axis=1))
            length=np.where(length<0.01,0.1,length)
            delta_pos=np.transpose(np.transpose(displacement)*t/length)
            if fixed is not None:
                # don't change positions of fixed nodes
                delta_pos[fixed]=0.0
            pos+=delta_pos
            # cool temperature
            t-=dt
            ###pos=_rescale_layout(pos)
        return pos
