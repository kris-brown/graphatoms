# External Modules
from typing import Tuple,Dict,List
import networkx as nx #type: ignore
import networkx.algorithms.isomorphism as iso #type: ignore
import ase #type: ignore
import collections

# Internal Modules
from graphatoms.misc.utilities import flatten  #type: ignore
from graphatoms.misc.atoms     import nonmetals,angle,dihedral,match_indices  #type: ignore
from graphatoms.structure.graph import GraphMaker,GraphInput  #type: ignore

"""
Useful functions for manipulating graph represenations of Atoms objects
"""

################################################################################
def graph_summary(G     : nx.Graph
                 ,name  : str       = ''
                 ) -> None:
    """
    Summarize some high level features of a graph (print to screen)
    """
    n = G.number_of_nodes()
    print('\nsummarizing Graph %s with %d nodes:'%(name,len(G.nodes())))
    print('# of nodes is ',n)
    print('# of edges  is ',G.number_of_edges())
    print('degree  is ',G.degree())
    edict = collections.defaultdict(int) # type: dict
    for i in range(n):
        for j in range(n):
            try: edict[(i,j)]+= len(G.edge[i][j])
            except KeyError: pass

    print(' ||| '.join([' %s : %d '%(k,v) for k,v in sorted(edict.items()) if v > 0]))

def get_angles(g : nx.Graph
              ) -> Dict[Tuple[int,int],float]:
    """
    Consider all pairs of connected edges.
    Return dictionary with keys being (fromNode,toNode) pairs
        and values being the angle between them
    """
    angles = {}
    for i,j,d in g.edges(data=True):
        for m,n,d2 in g.edges(data=True):
            if j == m: #connected
                angles[(d,d2)]=angle(d['vector'],d2['vector'])
    return angles

def get_dihedrals(g : nx.Graph
                 ) -> Dict[Tuple[int,int,int],float]:
    """
    Consider all triples of connected edges.
    Return dictionary with keys being (fromNode,toNode) pairs
        and values being the dihedral angle between them
    """
    dihedrals = {}
    for i,j,d in g.edges(data=True):
        for m,n,d2 in g.edges(data=True):
            for a,b,d3 in g.edges(data=True):
                if j == m and n == b: #connected
                    dihedrals[(d,d2,d3)] = dihedral(d['vector']
                                                  ,d2['vector']
                                                  ,d3['vector'])
    return dihedrals

def is_isomorphic(gi  : GraphInput
                 ,gi2 : GraphInput = None
                 ,gm  : GraphMaker = None
                 ,gm2 : GraphMaker = None
                 ) -> Tuple[bool,nx.Graph,nx.Graph]:
    """
    Returns a triple (Bool,Graph,Graph), where bool = nx.is_isomorphic
    """

    # Handle default arguments
    if gi2 is None:
        gi2 = gi
        assert gm2 is not None # we are varying the graphmaker, not the Atoms object

    if gm is None:
        gm = GraphMaker()      # use the default graphmaker
        assert gi2 is not None # we are varying the atoms object, not the graphmaker

    if gm2 is None:
        gm2 = gm # we are varying the atoms object, not the graphmaker

    # Construct graphs
    g1 = gm.make_graph(gi)
    g2 = gm2.make_graph(gi2)

    # Calculate if isomorphic
    symb_match  = iso.categorical_node_match('symbol','')
    return (nx.is_isomorphic(g1,g2,symb_match),g1,g2)

def reorder_nodes(graph         : nx.Graph
                 ,mapping_dict  : Dict[int,int]
                 ) -> nx.Graph:
    """
    Given a graph and mapping dictionary, reorder nodes and return new graph
    """
    R = nx.MultiGraph(cell=graph.graph['cell'])

    for i,index in mapping_dict.items(): # add nodes in correct order
        node = graph.node[index]
        R.add_node(i,symbol=node['symbol'],position=node['position']
                    ,magmom=node['magmom'],index=node['index'])

    for i,j,edge_dict in graph.edges(data=True):
        R.add_edge(mapping_dict[i], mapping_dict[j],weight=edge_dict['weight'],
                    pbc_shift=edge_dict['pbc_shift'])
    return R


def get_graph_diff(gi  : GraphInput
                  ,gi2 : GraphInput
                  ,gm  : GraphMaker
                  ,gm2 : GraphMaker
                  ) -> Tuple[nx.Graph,nx.Graph]:
    """
    Returns two graphs, the edges in G1 not present in G2 and vice versa
    """
    # Handle default arguments
    if gi2 is None:
        gi2 = gi
        assert gm2 is not None # we are varying the graphmaker, not the Atoms object

    if gm is None:
        gm = GraphMaker()      # use the default graphmaker
        assert gi2 is not None # we are varying the atoms object, not the graphmaker

    if gm2 is None:
        gm2 = gm # we are varying the atoms object, not the graphmaker

    symbs,symbs2 = [x.atoms.get_chemical_symbols() for x in [gi,gi2]]

    assert sorted(symbs)==sorted(symbs2) # same stoich required

    g1 = gm.make_graph(gi)
    g2 = gm2.make_graph(gi2)
    mapping_dict = match_indices(gi.atoms,gi2.atoms) #  tranformation to atoms1 to make it into atoms2
    g1_ = reorder_nodes(g1,mapping_dict)
    G1 = graph_diff(g1_,g2)
    G2 = graph_diff(g2,g1_)

    return G1,G2


def graph_diff(G : nx.Graph
              ,H : nx.Graph
              ) -> nx.Graph:
    """
    Edges present in G but NOT in H
    """
    R = nx.MultiGraph(cell=G.graph['cell'])
    R.add_nodes_from(G)
    h_edges = H.copy().edges()
    for i,j,d in G.edges(data=True):
        if (i,j) in h_edges: h_edges.remove((i,j))
        else: R.add_edges_from([(i,j)],**d)
    return R

def old_graph_diff(G : nx.Graph
                  ,H : nx.Graph
                  ) -> nx.Graph:
    """
    Edges present in G but NOT in H
    WHAT IS THE DIFFERENCE WITH GRAPH_DIFF?
    """
    R = nx.MultiGraph(cell=G.graph['cell'])
    R.add_nodes_from(G)
    def extract(tup : Tuple[int,int,dict])->Tuple[int,int,Tuple[int,int,int]]:
        (i,j,d) = tup
        return (i,j,d['pbc_shift'])  # essential Info
    edges_h = map(extract,H.edges(data=True))
    for x in G.edges(data=True):
        if extract(x) not in edges_h:
            i,j,d = x
            R.add_edges_from([(i,j)],**d)
    return R

def make_subgraph(graph      : nx.Graph
                 ,indexlist  : List[int]
                 ,pathlength : int
                 ) -> nx.Graph:
    """
    Take all edges within a certain pathlength and cut a subgraph
    """

    indices = set(indexlist)

    for _ in range(pathlength):
        indices.update(flatten([graph.edge[i].keys() for i in indices]))

    return graph.subgraph(indices)

def metal_independent(g : nx.Graph) -> nx.Graph:
    """
    Map all element numbers to 0 if they are metal atoms
    """
    h = g.copy()
    for n in h.nodes():
        if h.node[n]['symbol'] not in nonmetals:
            h.node[n]['symbol'] = 0
    return h

def is_subgraph(big           : nx.Graph
               ,small         : nx.Graph
               ,ignore_metals : bool        = False
               )->bool:
    """
    Helpful docstring
    """
    if ignore_metals: big,small = map(metal_independent,[big,small])

    symb_match  = iso.categorical_node_match('symbol','')
    gm          = iso.GraphMatcher(big,small,node_match=symb_match)
    return gm.subgraph_is_isomorphic()

def to_atoms(g : nx.Graph) -> ase.Atoms:
    a = ase.Atoms(cell=g.graph['cell'])
    for n in g.nodes():
        d = g.node[n]
        a.append(ase.Atom(d['symbol'],d['position']))
    return a
