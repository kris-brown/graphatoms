# External Modules
from typing import Iterator,List,Tuple,Dict,Callable
import math,os,copy,sys,json,collections
import networkx as nx   #type: ignore
import numpy as np      #type: ignore
import PyBliss as pb  #type: ignore
import ase.io           #type: ignore
from ase.data import chemical_symbols  #type: ignore
from ase.neighborlist import NeighborList  #type: ignore
# Internal Modules
from graphatoms.misc.utilities import (replacer,identity,get_analysis_folder,get_sherlock_folder,negate,true)  #type: ignore
from graphatoms.misc.utilities import flatten
from graphatoms.misc.atoms     import match_indices,angle,dihedral   #type: ignore

"""
Functions for creating and manipulating graph representations of Atoms objects
functions:
    graph_diff
    graph_summary
    graph_to_pybliss
classes:
    Edge
    GraphMaker
    GraphInput

"""
##############################################################################
class GraphInput(object):
    """
    TWO kinds of input for a graph:
        - stordir + trajname pair (it is expected that bonds.json exists)
        - ase.Atoms object (can only be used to construct jmol graphs)

    In the former case, we often need the Atoms object too so we load that here
    """
    def __init__(self
                ,stordir  : str
                ,trajname : str
                ,atoms    : ase.Atoms = None
                ) -> None:

        onlytraj = stordir is None

        self.onlytraj = onlytraj
        self.stordir  = stordir
        self.trajname = trajname

        if onlytraj:
            assert trajname is None
            assert isinstance(atoms,ase.Atoms)
            self.atoms    = atoms
        else:
            assert atoms is None
            # trajname can still be none, as some methods only need a directory

            pth = os.path.join(stordir,'%s.traj'%trajname)
            self.atoms = ase.io.read(get_sherlock_folder(pth))

    def __str__(self)->str:
        return str(self.__dict__)

    def __repr__(self)->str:
        return str(self)

    def view_atoms(self) -> None:
        """
        Opens an ASE gui session to view atoms of graph input
        """
        from ase.visualize import view  #type: ignore
        view(self.atoms)


##############################################################################
class Edge(object):
    """
    Helpful docstring
    """
    def __init__(self
                ,fromNode  : int
                ,toNode    : int
                ,distance  : float
                ,offset    : List[int]
                ,bondorder : float
                ,vector    : np.array  = None
                ) -> None:

        self.fromNode   = fromNode
        self.toNode     = toNode
        self.weight     = distance
        self.pbc_shift  = [int(x) for x in (offset[0],offset[1],offset[2])]
        self.bondorder  = bondorder
        self.vector     = vector

    def __str__(self)->str:
        return "<Edge(%d,%d,%.2f A,%s,bo: %.2f)>"%(
                self.fromNode,self.toNode,self.weight,self.pbc_shift
                ,self.bondorder)

    def __repr__(self)->str:
        return str(self)

    def add_edge_to_graph(self,grph : nx.Graph)->None:
        """
        Takes a networkx.Graph object, adds itself to it
        """
        prop_dict = {k: self.__dict__[k] for k in ('weight', 'pbc_shift', 'bondorder')}
        tupl = (self.toNode,self.fromNode)
        for (i,j,d) in grph.edges(data=True):
            if ((i,j),d['pbc_shift'])==(tupl,map(negate,self.pbc_shift)):
                return None # REPEAT EDGE, DON'T ADD
        grph.add_edges_from([tupl],**prop_dict)

    def inds_pbc(self)->Tuple[int,int,List[int]]:
        return (self.fromNode,self.toNode,self.pbc_shift)

##############################################################################

class GraphMaker(object):
    """
    Interface between Chargemol output (bond.json) and graph representations
    """
    def __init__(self
                , include_frac : float = 0.8
                , group_cut    : float = 0.3
                , min_bo       : float = 0.03
                , jmol         : float = False
                ) -> None:

        self.include_frac   = include_frac
        self.group_cut      = group_cut
        self.min_bo         = min_bo
        self.colored        = True
        self.jmol           = 0.01 if jmol==True else jmol # give a FLOAT to specify jmol tolerance

    def _get_edge_dicts(self,gi : GraphInput) -> List[dict]:
        """
        Finds bonds.json, loads it as a list of dictionaries
        """
        jsonpth = os.path.join(get_analysis_folder(gi.stordir)
                              ,'chargemol_analysis/%s/bonds.json'%gi.trajname)

        with open(jsonpth,'r') as f:
            return json.load(f)

    def _make_edges(self,gi : GraphInput) ->Dict[int,List[Edge]]:
        """
        Takes dejson'd bonds.json and makes Edge objects from the list
        """
        edges = collections.defaultdict(list) # type: dict

        if self.jmol:
            jmol_output = jmol(gi.atoms,self.jmol)
            for e in jmol_output:
                edges[e.fromNode].append(e)
            return edges
        else:
            dicts = self._get_edge_dicts(gi)
            keys  = ['bondorder','fromNode','toNode','offset']

            for d in dicts:
                bo,i,j,o = [d[x] for x in keys]

                if bo > self.min_bo:
                    p2 = gi.atoms[j].position + np.dot(o,gi.atoms.get_cell())
                    p1 = gi.atoms[i].position
                    dist  = np.linalg.norm(p2-p1)

                    edges[i].append(Edge(i,j,dist,np.array(o),bo)) # defaultdict useful
            return edges

    def _make_adjacency_matrix(self,gi : GraphInput) -> np.array:
        """
        Returns an n_atoms by n_atoms matrix where each row column pair contains
        the sum of the bond orders for the edges between those  two atoms
        """
        edge_dict = self._make_edges(gi)

        output = np.zeros((len(gi.atoms),len(gi.atoms)))

        for i,edges in edge_dict.items():
            for edge in edges:
                output[i,edge.toNode] += edge.bondorder

        return output

    def _make_group(self
                   ,edge_list : List[Edge]
                   ,group_cut : float       = None
                   ) -> List[List[Edge]]:
        """
        Partitions a set of Edges into groups with similar bond strength
        """
        from scipy.cluster.hierarchy import fclusterdata #type: ignore

        # Handle edge cases
        if len(edge_list)==0:  return []
        if len(edge_list)==1:  return [edge_list]

        if group_cut is None: group_cut = self.group_cut

        strs = np.array([[e.bondorder for e in edge_list]]).T # create (n_edges,1) array
        groups = collections.defaultdict(list)  #type: dict

        group_inds = list(fclusterdata(strs,group_cut
                                ,criterion='distance',method='ward'))

        for i in range(len(edge_list)): groups[group_inds[i]].append(edge_list[i])
        maxbo_groups = [(max([e.bondorder for e in es]),es) for es in groups.values()]
        sorted_maxbo_groups = list(reversed(sorted(maxbo_groups)))
        return [es for maxbo,es in sorted_maxbo_groups]


    def _get_filtered_edges(self
                           ,gi : GraphInput
                           ) -> List[Edge]: # BOA
        """
        Return a filtered subset of the edges serialized in bonds.json
        """

        edges      = self._make_edges(gi)
        edge_vals  = flatten(list(edges.values()))

        output = [] # type: List[Edge]
        if self.jmol:
            for e in  edge_vals:    # only need to filter duplicate edges
                neg_pbc = [-x for x in e.pbc_shift]
                dup_e = Edge(e.toNode,e.fromNode,e.weight,neg_pbc,e.bondorder)
                if dup_e not in output:
                    output.append(e)
            return output
        else:
            total_bo = {ind:sum([e.bondorder for e in es]) for ind,es in edges.items()}
            groups   = {ind: self._make_group(es) for ind,es in edges.items()}

            max_ind  = max([max(e.fromNode, e.toNode) for e in edge_vals])

            for ind in range(max_ind+1):
                accumulated = 0.0
                for i, group in enumerate(groups[ind]):
                    output.extend(group)
                    accumulated += sum([e.bondorder for e in group])/total_bo[ind]
                    if accumulated > self.include_frac:
                        break
            return output

    def _graph_to_pybliss(self,gi : GraphInput) -> pb.Graph:
        """
        Converts undirected multigraph into undirected graph by replacing n edges
            between v1 and v2 with n new nodes bridging v1 and v2
        """
        def edge_add(pb,tup,n):
            (v1,v2) = tup
            def add_new_node(pb):
                """adds a new node to pybliss graph, returns its value"""
                max_node = max(pb.get_vertices())
                pb.add_vertex(max_node+1,0)
                return max_node+1

            if n == 1: pb.add_edge(v1,v2)
            else:
                for i in range(1,n):
                    new_node = add_new_node(pb)
                    pb.add_edge(v1,new_node)
                    pb.add_edge(new_node,v2)

        g  = self.make_graph(gi)
        pbg = pb.Graph()
        for i in range(len(g.nodes())):
            symb = g.node[i]['symbol'] #0 reserved for dummy nodes
            n = g.nodes()[i]
            j = chemical_symbols.index(symb)
            pbg.add_vertex(n,j)

        edge_dict = {x:len([y for y in g.edges() if str(y) == str(x)])
                                                    for x in g.edges()}
        for e,n in edge_dict.items():
            edge_add(pbg,e,n)

        return pb

    def _get_sum_of_bond_order_data(self
                                   ,gi: GraphInput
                                   ,show_indices : List[int] = None
                                   )->Tuple[List[str],np.array]:
        """
        Sum of bond orders
        """

        adj_matrix         = self._make_adjacency_matrix(gi)
        sum_of_bond_orders = np.sum(adj_matrix, axis = 1)

        if show_indices is None:
            show_indices = list(range(adj_matrix.shape[0]))

        indices =  [range(len(gi.atoms))[i] for i in show_indices]
        labels = ['%s-%s-%d'%(gi.trajname,atom.symbol,atom.index) for atom in gi.atoms if atom.index in indices]
        return (labels, sum_of_bond_orders[indices])



    def plot_plotly_atoms(self, gi : GraphInput) -> nx.Graph:
        """
        make a plotly atoms
        """
        import graphatoms.structure.plotly_atoms as mp  #type: ignore
        graph = self.make_graph(gi)
        return mp.PlotlyAtoms(graph).plot()

    def many_plotly_atoms(self, gi: GraphInput) -> None:
        """
        make many of them
        """
        import graphatoms.structure.plotly_atoms as mp #type: ignore
        trajs = os.listdir(os.path.join(gi.stordir,'chargemol_analysis'))
        folder = gi.stordir.split('/')[-1]
        for trajname in sorted(trajs):
            temp_gi = GraphInput(gi.stordir,trajname)
            graph = self.make_graph(temp_gi)
            mp.PlotlyAtoms(graph).plot(file_name = '%s_%s.html'%(folder,trajname),offline=False)

    def plot_bond_order_analysis(self
                                ,gi          : GraphInput
                                ,show_groups : bool                      = True
                                ,filt        : Callable[[ase.Atom],bool] = true
                                ,show        : bool                      = True
                                ,atoms       : ase.Atoms                 = None
                                ) -> None:
        """
        plots bond order analysis maybe?
        """
        import matplotlib; matplotlib.use('Qt4Agg') #type: ignore
        import matplotlib.pyplot as plt             #type: ignore

        def get_symb(z):   return gi.atoms[z].symbol

        edges    = self._make_edges(gi) # {index:list of Edges from index}
        groups   = {ind: self._make_group(es) for ind,es in edges.items()}
        total_bo = {ind:sum([e.bondorder for e in es]) for ind,es in edges.items()}

        f,ax  = plt.subplots(nrows=1,ncols=1)
        plt.subplots_adjust(bottom=0.2)
        xmin,height   = 0, 0.2

        def align(h,v): # creates keyword dictionary for ax.text
            d = {'t':'top','b':'bottom','c':'center','l':'left','r':'right'}
            return {'horizontalalignment':d[h],'verticalalignment':d[v]}

        for ind in range(len(gi.atoms)):
            if filt(atoms[ind]):
                xmax   = max([e.bondorder for e in edges[ind]])
                ax.hlines(ind,xmin,xmax)
                ax.vlines(xmin,ind-height,ind+height)
                ax.vlines(xmax,ind-height,ind+height)
                ax.text(0,ind,get_symb(ind)+' ',weight='bold',**align('r','c'))

                if show_groups and not self.jmol: # plot Group Information
                    accumulated = 0. # reset counter for bond strength accumulation
                    n_groups    = len(groups[ind])
                    for group_index,group in enumerate(groups[ind]):
                        over_thres  = accumulated > self.include_frac
                        weight,size = ('bold',10) if not over_thres else ('light',7)
                        lastgroup   = group_index == n_groups - 1
                        accumulated+= sum([e.bondorder for e in group])/total_bo[ind]
                        accum_txt   = '' if lastgroup else ' (%d%%)'%(accumulated*100)

                        mean = np.mean([e.bondorder for e in group])
                        ax.text(mean,ind+height, str(group_index) + accum_txt
                                ,size = size ,weight = weight, **align('c','b'))

                # make smaller groups for plotting atoms along the line
                # we have to treat each toNode index separately to not lose information
                elems = list(set([get_symb(edg.toNode) for edg in edges[ind]]))
                for elem in elems:
                    elem_edges = [edg for edg in edges[ind] if get_symb(edg.toNode) == elem]

                    smallgroups = self._make_group(elem_edges,0.03)

                    for smallgroup in smallgroups:
                        n = len(smallgroup) #multiplicity
                        s = np.mean([e.bondorder for e in smallgroup])
                        mult_txt = '(x%d)'%n if n > 1 else ''
                        ax.text(s,ind,elem,size=12,**align('c','c'))
                        ax.plot(s,ind,'ro',ms=15,mfc='r')
                        ax.text(s,ind-0.3,mult_txt,size=10,**align('c','t'))

        vtitle = 'jmol analysis with tol = %.3f'%self.jmol
        ctitle = ('Chargemol Bond analysis for %s: group_cut = '%gi.trajname
                 +'%.2f'%(self.group_cut))
        ax.set_title(vtitle if self.jmol else ctitle)
        if show: plt.show()

    def many_bond_order_analysis(self, gi : GraphInput) -> None:
        """
        Plot many of 'em
        """
        import graphatoms.structure.plotly_atoms as mp  #type: ignore
        import matplotlib.pyplot as plt
        trajs = os.listdir(os.path.join(gi.stordir,'chargemol_analysis'))
        for trajname in sorted(trajs):
            temp_gi = GraphInput(gi.stordir,trajname)
            self.plot_bond_order_analysis(temp_gi
                                         ,filt = lambda x: x.symbol=='N'
                                         ,show = False)
        plt.show()

    def make_graph(self, gi : GraphInput) -> nx.Graph:
        """
        Create NetworkX Graph Object
        """

        adj_matrix = self._make_adjacency_matrix(gi)
        G          = nx.MultiGraph(cell = gi.atoms.get_cell()
                                  ,adj_matrix = adj_matrix)  # Initialize graph

        for i in range(len(gi.atoms)): # add nodes
            G.add_node(i,symbol   = gi.atoms[i].symbol if self.colored else 1
                        ,position = gi.atoms[i].position
                        ,index    = i
                        ,magmom   = gi.atoms[i].magmom)

        edges = self._get_filtered_edges(gi)
        for e in edges: e.add_edge_to_graph(G)
        return G

    def make_unweighted_canonical(self, gi : GraphInput) -> str:
        """
        We shouldn't have to ever do nx.is_isomorphic(GRAPH1,GRAPH2,symb_match)
        Because there is a canonical form for unweighted multigraphs
        Thus if we store the normalized string in the database, we can just test
          for string equality.
        But to compare atoms using dist_match, one needs nx.is_isomorphic

        Return string characterizing connectivity (not weights)
        """
        pb = self._graph_to_pybliss(gi)
        return str(pb.relabel(pb.canonical_labeling()))

    def plot_many_sum_of_bond_orders(self
                                    ,gi         : GraphInput
                                    ,show_index : int
                                    ) -> None:
        """
        Plots many of them
        """
        import matplotlib; matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        trajs = os.listdir(os.path.join(gi.stordir,'chargemol_analysis'))

        #Get the data and labels for each chargemol_analysis in stordir
        labels, sum_of_bond_orders = [], [] # type: Tuple[List[str],List[float]]
        for trajname in sorted(trajs):
            temp_gi = GraphInput(gi.stordir,trajname)
            label_curr, sum_of_bond_order_curr = self._get_sum_of_bond_order_data(temp_gi, show_indices = [show_index])
            labels += label_curr; sum_of_bond_orders += sum_of_bond_order_curr.tolist()

        #Plotting
        fig, ax = plt.subplots()
        ax.barh(range(len(trajs)), sum_of_bond_orders,[0.5]*len(sum_of_bond_orders))
        ax.set_yticks(range(len(sum_of_bond_orders)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Bond Order')
        plt.show()


    def show_adj_heatmap(self,gi:GraphInput) -> None:
        """
        Heatmap representation of bonding between atoms (ignores pbc)
        """
        import matplotlib; matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        f,ax  = plt.subplots(nrows=1,ncols=1)
        data  = self._make_adjacency_matrix(gi)

        heatmap = ax.pcolor(data,cmap='seismic')
        plt.colorbar(heatmap)

        ax.set_title('Adjacency Matrix for %s'%(gi))
        ax.set_yticklabels(gi.atoms.get_chemical_symbols())
        ax.set_xticklabels(gi.atoms.get_chemical_symbols())

        plt.show()

########################################################################
########################################################################

def jmol(atoms : ase.Atoms
        ,tol   : float      = 0.01
        ) -> List[Edge]:
    """Take an Atoms object and return list of Edges (no bonds.json needed)"""

    from pymatgen.analysis.structure_analyzer import JMolCoordFinder  #type: ignore
    from pymatgen.io.ase import AseAtomsAdaptor  #type: ignore
    positions = atoms.get_positions()
    floor     = np.vectorize(math.floor)

    def get_index_edges(ind):
        p         = positions[ind]
        pmgatoms  = AseAtomsAdaptor().get_structure(atoms)
        pmg_sites = JMolCoordFinder().get_coordinated_sites(pmgatoms,ind,tol=tol)

        def get_ind(s): return np.argmin(map(np.linalg.norm,positions - s.to_unit_cell.coords))
        def get_pbc(s): return floor(s.frac_coords)
        def get_v(s):   return np.linalg.norm(s.coords - p)
        return [Edge(ind,get_ind(s),get_v(s),get_pbc(s),5-get_v(s),0) for s in pmg_sites]

    return flatten([get_index_edges(i) for i in range(len(atoms))])


########################################################################

######
# Test
#-----
def example_graph(i : int = 0)-> nx.Graph:
    gm = GraphMaker(min_bo=0.05,include_frac=0.9)
    f  = lambda x: gm.make_graph(GraphInput(x,'init'))
    graphs = map(f,['/scratch/users/ksb/share/jobs/ksb/151034096235/'  # Os hcp
                     ,'/scratch/users/ksb/share/jobs/ksb/150959874894/' # Pd fcc
                     ,'/scratch/users/ksb/share/jobs/ksb/150961514721/' # Ru hcp
                     ,'/scratch/users/ksb/share/jobs/ksb/150958374384/' # Ba bcc
                     ])
    return list(graphs)[i]

def adsorbate_graphs(i : int = 0)-> Tuple[nx.Graph,List[int]]:
    """
    Returns a pair (Graph,List of adsorbate indices])
    """
    gm = GraphMaker(min_bo=0.05,include_frac=0.8)
    f  = lambda x:  (gm.make_graph(GraphInput(x[0],'init')),x[1])
    graphs = map(f,[('/nfs/slac/g/suncatfs/ksb/share/jobs/mstatt/151035187731/',[40])    # H
                 ,('/nfs/slac/g/suncatfs/ksb/share/jobs/mstatt/151036038481/',[40,41]) # NN
                 ,('/nfs/slac/g/suncatfs/ksb/share/jobs/mstatt/151033093931/',[40,41]) # OH
                 ,('/scratch/users/ksb/share/jobs/mstatt/151055130379/',[39,40])       # vac NN
               ])
    return list(graphs)[i]


def test_subgraph(p : int = 1)-> None:
    from graphatoms.structure.graph_utils import make_subgraph,graph_summary,is_subgraph  #type: ignore
    g = example_graph()
    h = example_graph(1)
    i = example_graph(2)
    j = example_graph(3)
    g2 = make_subgraph(h,[0,1],p)
    graph_summary(g2)
    g3 = make_subgraph(i,[0,1],p)
    graph_summary(g3)
    g4 = make_subgraph(j,[0,1],p)
    graph_summary(g4)

    print(is_subgraph(g,g2,ignore_metals=True))
    print(is_subgraph(g,g3,ignore_metals=True))
    print(is_subgraph(g,g4,ignore_metals=True))

def test_adsorbate(i : int) -> None:
    from graphatoms.structure.graph_utils import make_subgraph,graph_summary,is_subgraph,to_atoms #type: ignore
    from ase.visualize import view #type: ignore
    g,ads = adsorbate_graphs(i)
    g2 = make_subgraph(g,ads,1)
    graph_summary(g2)
    view(to_atoms(g))
