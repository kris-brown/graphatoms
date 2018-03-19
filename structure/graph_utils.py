# External Modules
import networkx as nx
import networkx.algorithms.isomorphism as iso
import ase,collections

# Internal Modules
from CataLog.misc.utilities import flatten
from CataLog.misc.atoms     import nonmetals
################################################################################
def graph_summary(G,name=''):
    n = G.number_of_nodes()
    print '\nsummarizing Graph %s with %d nodes:'%(name,len(G.nodes()))
    print '# of nodes is ',n
    print '# of edges  is ',G.number_of_edges()
    print 'degree  is ',G.degree()
    edict = collections.defaultdict(int)
    for i in range(n):
        for j in range(n):
            try: edict[(i,j)]+= len(G.edge[i][j])
            except KeyError: pass

    print ' ||| '.join([' %s : %d '%(k,v) for k,v in sorted(edict.items()) if v > 0])

def get_angles(g):
    angles = {}
    for i,j,d in g.edges(data=True):
        for m,n,d2 in g.edges(data=True):
            if j == m: #connected
                angles[(d,d2)]=angle(d['vector'],d2['vector'])
    return angles

def get_dihedrals(g):
    dihedrals = {}
    for i,j,d in g.edges(data=True):
        for m,n,d2 in g.edges(data=True):
            for a,b,d3 in g.edges(data=True):
                if j == m and n == b: #connected
                    dihedrals[(d,d2,d3)]=dihedral(d['vector'],d2['vector'],d3['vector'])
    return dihedrals

def is_isomorphic(stordir,trajname='final',stordir2=None,trajname2=None
                    ,graphmaker=None,graphmaker2=None):
    """Returns a triple (Bool,Graph,Graph), where bool = nx.is_isomorphic"""
    if stordir2 is None: stordir2 = stordir
    if trajname2 is None: trajname2 = trajname
    if graphmaker is None: graphmaker = GraphMaker()
    if graphmaker2 is None: graphmaker2 = graphmaker # by default use same one
    g1 = graphmaker.make_graph(stordir,trajname)
    graph_summary(g1)
    g2 = graphmaker2.make_graph(stordir,trajname2)
    graph_summary(g2)
    symb_match     = iso.categorical_node_match('symbol','')
    return (nx.is_isomorphic(g1,g2,symb_match),g1,g2)

def reorder_nodes(graph,mapping_dict):
    R = nx.MultiGraph(cell=graph.graph['cell'])

    for i,index in mapping_dict.items(): # add nodes in correct order
        node = graph.node[index]
        R.add_node(i,symbol=node['symbol'],position=node['position']
                    ,magmom=node['magmom'],index=node['index'])

    for i,j,edge_dict in graph.edges(data=True):
        R.add_edge(mapping_dict[i], mapping_dict[j],weight=edge_dict['weight'],
                    pbc_shift=edge_dict['pbc_shift'])
    return R


def get_graph_diff(stordir,trajname='final',stordir2=None,trajname2=None
                    ,graphmaker=None,graphmaker2=None):
    """Returns two graphs, the edges in G1 not present in G2 and vice versa"""
    if stordir2 is None: stordir2 = stordir
    if trajname2 is None: trajname2 = trajname
    if graphmaker is None: graphmaker = GraphMaker()
    if graphmaker2 is None: graphmaker2 = graphmaker # by default use same one

    atoms = graphmaker.make_atoms(stordir,trajname)
    atoms2 = graphmaker.make_atoms(stordir2,trajname2)
    assert sorted(atoms.get_chemical_symbols())==sorted(atoms2.get_chemical_symbols()) # same stoich required

    g1 = graphmaker.make_graph(stordir,trajname)
    if True: graph_summary(g1,'g1')
    g2 = graphmaker2.make_graph(stordir2,trajname2)
    if True: graph_summary(g2,'g2')
    mapping_dict = match_indices(atoms,atoms2) #  tranformation to atoms1 to make it into atoms2
    g1_ = reorder_nodes(g1,mapping_dict)
    if True: graph_summary(g1_,'g1_')
    G1 = graph_diff(g1_,g2)
    if True: graph_summary(G1,'G1')
    G2 = graph_diff(g2,g1_)
    if True: graph_summary(G2,'G2')

    return G1,G2


def graph_diff(G,H):
    """Edges present in G but NOT in H"""
    R = nx.MultiGraph(cell=G.graph['cell'])
    R.add_nodes_from(G)
    h_edges = H.copy().edges()
    for i,j,d in G.edges(data=True):
        if (i,j) in h_edges: h_edges.remove((i,j))
        else: R.add_edges_from([(i,j)],**d)
    return R

def old_graph_diff(G,H):
    """Edges present in G but NOT in H"""
    R = nx.MultiGraph(cell=G.graph['cell'])
    R.add_nodes_from(G)
    def extract((i,j,d)): return (i,j,d['pbc_shift'])  # essential Info
    edges_h = map(extract,H.edges(data=True))
    for x in G.edges(data=True):
        if extract(x) not in edges_h:
            i,j,d = x
            R.add_edges_from([(i,j)],**d)
    return R

def make_subgraph(graph,index,pathlength):
    """Take all edges within a certain pathlength and cut a subgraph"""

    if isinstance(index,list): indices = set(index)
    else:                      indices = set([index])

    for _ in range(pathlength):
        indices.update(flatten([graph.edge[i].keys() for i in indices]))

    return graph.subgraph(indices)

def metal_independent(g):
    """Map all element numbers to 0 if they are metal atoms"""
    h = g.copy()
    for n in h.nodes():
        if h.node[n]['symbol'] not in nonmetals:
            h.node[n]['symbol'] = 0
    return h

def is_subgraph(big,small,ignore_metals=False):
    if ignore_metals: big,small = map(metal_independent,[big,small])

    symb_match  = iso.categorical_node_match('symbol','')
    gm          = iso.GraphMatcher(big,small,node_match=symb_match)
    return gm.subgraph_is_isomorphic()

def to_atoms(g):
    a = ase.Atoms(cell=g.graph['cell'])
    for n in g.nodes():
        d = g.node[n]
        a.append(ase.Atom(d['symbol'],d['position']))
    return a
