import networkx as nx
from networkx_viewer import Viewer

G = nx.MultiGraph()
edges = ([('e0', 'e1', 0), ('e0', 'e1', 1), ('e0', 'e2', 0), ('e0', 'e2', 1), ('e0', 'e3'), ('e0', 'e4'), ('e0', 'e5'),
          ('e0', 'e7'), ('e0', 'e8'),
          ('e1', 'e2'), ('e1', 'e4'),
          ('e2', 'e3'),
          ('e3', 'e4'),
          ('e4', 'e5'), ('e4', 'e6'),
          ('e5', 'e6'), ('e5', 'e7'), ('e5', 'e8', 0), ('e5', 'e8', 1),
          ('e6', 'e7'),
          ('e7', 'e8')])

G.add_edges_from(edges)
app = Viewer(G)
app.mainloop()