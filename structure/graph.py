# External Modules
import networkx as nx
import numpy as np
import math,ase.io,os,copy,sys,json,collections
from ase.data import chemical_symbols
from ase.neighborlist import NeighborList
from PyBliss import Graph
# Internal Modules
from CataLog.misc.utilities import (replacer,identity,get_analysis_folder
                                   ,get_sherlock_folder,negate,true)
from CataLog.misc.utilities import flatten
from CataLog.misc.atoms     import match_indices,angle,dihedral
##############################################################################
"""
Functions for creating and manipulating graph representations of Atoms objects
functions:
    graph_diff
    graph_summary
    graph_to_pybliss
classes:
    Edge
    GraphMaker

"""

class Edge(object):
    def __init__(self,fromNode,toNode,distance,offset,bondorder,vector=None):
        self.fromNode   = fromNode
        self.toNode     = toNode
        self.weight     = distance
        self.pbc_shift  = map(int,(offset[0],offset[1],offset[2]))
        self.bondorder  = bondorder
        self.vector     = vector

    def __str__(self): return "<Edge(%d,%d,%.2f A,%s,bo: %.2f)>"%(
                                self.fromNode,self.toNode,self.weight,self.pbc_shift
                                ,self.bondorder)

    def __repr__(self): return str(self)

    def add_edge_to_graph(self,grph):
        """Takes a networkx.Graph object, adds itself to it"""
        prop_dict = {k: self.__dict__[k] for k in ('weight', 'pbc_shift', 'bondorder')}
        tupl = (self.toNode,self.fromNode)
        for (i,j,d) in grph.edges(data=True):
            if ((i,j),d['pbc_shift'])==(tupl,map(negate,self.pbc_shift)):
                return None # REPEAT EDGE, DON'T ADD
        grph.add_edges_from([tupl],**prop_dict)

    def inds_pbc(self): return (self.fromInd,self.toInd,self.pbc_shift)

class GraphMaker(object):
    """
    Interface between Chargemol output (bond.json) and graph representations
    include_frac  :: Float
    group_cut     :: Float
    min_bo        :: Float
    jmol          :: Float or Bool
    """
    def __init__(self, include_frac=0.8,group_cut=0.3,min_bo=0.03,jmol=False):
        self.include_frac   = include_frac
        self.group_cut      = group_cut
        self.min_bo         = min_bo
        self.colored        = True
        self.jmol           = 0.01 if jmol==True else jmol # give a FLOAT to specify jmol tolerance

    def _get_edge_dicts(self,stordir,trajname):
        """Finds bonds.json, loads it as a list of dictionaries"""
        jsonpth = os.path.join(get_analysis_folder(stordir)
                              ,'chargemol_analysis/%s/bonds.json'%trajname)
        with open(jsonpth,'r') as f: return json.load(f)

    def _make_edges(self,stordir,trajname,atoms):
        """Takes dejson'd bonds.json and makes Edge objects from the list"""
        edges = collections.defaultdict(list)

        if self.jmol:
            assert atoms is not None
            jmol_output = jmol(atoms,self.jmol)
            for e in jmol_output:
                edges[e.fromNode].append(e)
            return edges
        else:
            atoms = self.make_atoms(stordir,trajname)
            dicts = self._get_edge_dicts(stordir,trajname)
            keys  = ['bondorder','fromNode','toNode','offset']
            for d in dicts:
                bo,i,j,o = [d[x] for x in keys]

                if bo > self.min_bo:
                    p2 = atoms[j].position + np.dot(o,atoms.get_cell())
                    p1 = atoms[i].position
                    d  = np.linalg.norm(p2-p1)

                    edges[i].append(Edge(i,j,d,np.array(o),bo)) # defaultdict useful
            return edges

    def _make_adjacency_matrix(self,stordir,trajname,atoms):
        """
        Returns an n_atoms by n_atoms matrix where each row column pair contains
        the sum of the bond orders for the edges between those  two atoms
        """
        edge_dict = self._make_edges(stordir,trajname,atoms)

        if atoms is None:
            atoms = self.make_atoms(stordir,trajname)

        output = np.zeros((len(atoms),len(atoms)))

        for i,edges in edge_dict.items():
            for edge in edges: output[i,edge.toNode] += edge.bondorder

        return output

    def _make_group(self,edge_list, group_cut = None):
        from scipy.cluster.hierarchy import fclusterdata
        if group_cut is None: group_cut = self.group_cut

        # Handle edge cases
        if not edge_list:      return edge_list
        if len(edge_list)==1:  return [edge_list]

        strs = np.array([[e.bondorder for e in edge_list]]).T # create (n_edges,1) array
        groups = collections.defaultdict(list)    # initialize group dict

        group_inds = list(fclusterdata(strs,group_cut
                                ,criterion='distance',method='ward'))

        for i in range(len(edge_list)): groups[group_inds[i]].append(edge_list[i])
        maxbo_groups = [(max([e.bondorder for e in es]),es) for es in groups.values()]
        sorted_maxbo_groups = list(reversed(sorted(maxbo_groups)))
        return [es for maxbo,es in sorted_maxbo_groups]


    def _get_filtered_edges(self,stordir,trajname,atoms): # BOA
        """
        Return a filtered subset of the edges serialized in bonds.json
        """

        output   = []                                   # initialize
        edges    = self._make_edges(stordir,trajname,atoms)   # {index:list of edges from index}
        if self.jmol:
            for e in  flatten(edges.values()):    # only need to filter duplicate edges
                dup_e = Edge(e.toNode,e.fromNode,e.weight,map(negate,e.pbc_shift),e.bondorder)
                if dup_e not in output: output.append(e)
            return output
        total_bo = {ind:sum([e.bondorder for e in es]) for ind,es in edges.items()}
        groups   = {ind: self._make_group(es) for ind,es in edges.items()}

        max_ind  = max([max(e.fromNode, e.toNode) for e in flatten(edges.values())])

        for ind in range(max_ind+1):
            accumulated = 0
            for i, group in enumerate(groups[ind]):
                output.extend(group)
                accumulated += sum([e.bondorder for e in group])/total_bo[ind]
                if accumulated > self.include_frac: break
        return output

    def _graph_to_pybliss(self,stordir,trajname,atoms=None):
        """
        Converts undirected multigraph into undirected graph by replacing n edges
            between v1 and v2 with n new nodes bridging v1 and v2
        """
        def edge_add(pb,(v1,v2),n):
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

        g  = self.make_graph(stordir,trajname,atoms)
        pb = Graph()
        for i in range(len(g.nodes())):
            symb = g.node[i]['symbol'] #0 reserved for dummy nodes
            n = g.nodes()[i]
            j = chemical_symbols.index(symb)
            pb.add_vertex(n,j)

        edge_dict = {x:len([y for y in g.edges() if str(y) == str(x)]) for x in g.edges()}
        for e,n in edge_dict.items(): edge_add(pb,e,n)
        return pb

    def _get_sum_of_bond_order_data(self, stordir, trajname = 'final', show_indices = None):
        """ """
        adj_matrix = self._make_adjacency_matrix(stordir, trajname)
        atoms_obj = self.make_atoms(stordir,trajname)
        sum_of_bond_orders = np.sum(adj_matrix, axis = 1)
        if show_indices is None:
            show_indices = range(adj_matrix.shape[0])

        indices =  [range(len(atoms_obj))[i] for i in show_indices]
        labels = ['%s-%s-%d'%(trajname,atom.symbol,atom.index) for atom in atoms_obj if atom.index in indices]
        return (labels, sum_of_bond_orders[indices])


    def make_atoms(self,stordir,trajname='final'):
        """HA HA HA"""
        pth = os.path.join(stordir,'%s.traj'%trajname)
        return ase.io.read(get_sherlock_folder(pth))

    def view_atoms(self,stordir,trajname='final'):
        from ase.visualize import view
        view(self.make_atoms(stordir,trajname))

    def plot_plotly_atoms(self, stordir, trajname = 'final'):
        """ """
        import misc.plotly_atoms as mp
        graph = self.make_graph(stordir,trajname)
        return mp.PlotlyAtoms(graph).plot()

    def many_plotly_atoms(self, stordir):
        """ """
        import misc.plotly_atoms as mp
        trajs = os.listdir(os.path.join(stordir,'chargemol_analysis'))
        folder = stordir.split('/')[-1]
        for trajname in sorted(trajs):
            graph = self.make_graph(stordir,trajname)
            mp.PlotlyAtoms(graph).plot(file_name = '%s_%s.html'%(folder,trajname),offline=False)

    def plot_bond_order_analysis(self,stordir,trajname='final'
                                ,show_groups=True,filt=true
                                ,show=True,atoms=None):
        """ """
        import matplotlib; matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt

        def get_symb(z):   return atoms[z].symbol

        edges    = self._make_edges(stordir,trajname,atoms) # {index:list of Edges from index}
        groups   = {ind: self._make_group(es) for ind,es in edges.items()}
        total_bo = {ind:sum([e.bondorder for e in es]) for ind,es in edges.items()}

        atoms = self.make_atoms(stordir, trajname)
        f,ax  = plt.subplots(nrows=1,ncols=1)
        plt.subplots_adjust(bottom=0.2)
        xmin,height   = 0, 0.2

        def align(h,v): # creates keyword dictionary for ax.text
            d = {'t':'top','b':'bottom','c':'center','l':'left','r':'right'}
            return {'horizontalalignment':d[h],'verticalalignment':d[v]}

        for ind in range(len(atoms)):
            if filt(atoms[ind]):
                xmax   = max([e.bondorder for e in edges[ind]])
                ax.hlines(ind,xmin,xmax)
                ax.vlines(xmin,ind-height,ind+height)
                ax.vlines(xmax,ind-height,ind+height)
                ax.text(0,ind,get_symb(ind)+' ',weight='bold',**align('r','c'))

                if show_groups and not self.jmol: # plot Group Information
                    accumulated = 0 # reset counter for bond strength accumulation
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
        ctitle = ('Chargemol Bond analysis for %s: group_cut = '%trajname
                 +'%.2f'%(self.group_cut))
        ax.set_title(vtitle if self.jmol else ctitle)
        if show: plt.show()

    def many_bond_order_analysis(self, stordir):
        """ """
        import misc.plotly_atoms as mp
        import matplotlib.pyplot as plt
        trajs = os.listdir(os.path.join(stordir,'chargemol_analysis'))
        folder = stordir.split('/')[-1]
        for trajname in sorted(trajs):
            self.plot_bond_order_analysis(stordir,trajname,filt=lambda x: x.symbol=='N',show=False)
        plt.show()

    def make_graph(self,stordir,trajname='final',atoms=None):
        """Create NetworkX Graph Object"""

        if atoms is None: # ONLY useful to provide if doing jmol analysis
            atoms = self.make_atoms(stordir,trajname)

        adj_matrix = self._make_adjacency_matrix(stordir,trajname,atoms)
        G          = nx.MultiGraph(cell = atoms.get_cell()
                                  ,adj_matrix = adj_matrix)  # Initialize graph

        for i in range(len(atoms)): # add nodes
            G.add_node(i,symbol=atoms[i].symbol if self.colored else 1
                        ,position=atoms[i].position
                        ,index = i,magmom=atoms[i].magmom)

        edges = self._get_filtered_edges(stordir,trajname,atoms)
        for e in edges: e.add_edge_to_graph(G)
        return G

    def make_unweighted_canonical(self,stordir=None,trajname = 'final',atoms=None):
        """
        We shouldn't have to ever do nx.is_isomorphic(GRAPH1,GRAPH2,symb_match)
        Because there is a canonical form for unweighted multigraphs
        Thus if we store the normalized string in the database, we can just test
          for string equality.
        But to compare atoms using dist_match, one needs nx.is_isomorphic

        Return string characterizing connectivity (not weights)
        """
        assert (stordir is None) ^ (atoms is None), 'weird input to make_unweighted_canonical'
        pb = self._graph_to_pybliss(stordir, trajname, atoms)
        return str(pb.relabel(pb.canonical_labeling()))

    def plot_many_sum_of_bond_orders(self, stordir, show_index):
        """ """
        import matplotlib; matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        trajs = os.listdir(os.path.join(stordir,'chargemol_analysis'))
        folder = stordir.split('/')[-1]

        #Get the data and labels for each chargemol_analysis in stordir
        labels, sum_of_bond_orders = [], []
        for trajname in sorted(trajs):
            label_curr, sum_of_bond_order_curr = self._get_sum_of_bond_order_data(stordir, trajname, show_indices = [show_index])
            labels += label_curr; sum_of_bond_orders += sum_of_bond_order_curr.tolist()

        #Plotting
        fig, ax = plt.subplots()
        ax.barh(range(len(trajs)), sum_of_bond_orders,[0.5]*len(sum_of_bond_orders))
        ax.set_yticks(range(len(sum_of_bond_orders)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Bond Order')
        plt.show()


    def show_adj_heatmap(self,stordir,trajname):
        """Heatmap representation of bonding between atoms (ignores pbc)"""
        import matplotlib; matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        f,ax  = plt.subplots(nrows=1,ncols=1)
        data  = self._make_adjacency_matrix(stordir,trajname)
        atoms = self.make_atoms(stordir,trajname)

        heatmap = ax.pcolor(data,cmap='seismic')
        plt.colorbar(heatmap)

        ax.set_title('Adjacency Matrix for %s (%s)'%(trajname,stordir))
        ax.set_yticklabels(atoms.get_chemical_symbols())
        ax.set_xticklabels(atoms.get_chemical_symbols())

        plt.show()

########################################################################
########################################################################

def jmol(atoms,tol=0.01):
    """Take an Atoms object and return list of Edges (no bonds.json needed)"""

    from pymatgen.analysis.structure_analyzer import JMolCoordFinder
    from pymatgen.io.ase import AseAtomsAdaptor
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
def example_graph(i=0):
    gm = GraphMaker(min_bo=0.05,include_frac=0.9)
    graphs = [gm.make_graph('/scratch/users/ksb/share/jobs/ksb/151034096235/','init') # Os hcp
             ,gm.make_graph('/scratch/users/ksb/share/jobs/ksb/150959874894/','init') # Pd fcc
             ,gm.make_graph('/scratch/users/ksb/share/jobs/ksb/150961514721/','init') # Ru hcp
             ,gm.make_graph('/scratch/users/ksb/share/jobs/ksb/150958374384/','init') # Ba bcc
             ]
    return graphs[i]

def adsorbate_graphs(i=0):
    gm = GraphMaker(min_bo=0.05,include_frac=0.8)
    graphs = [(gm.make_graph('/nfs/slac/g/suncatfs/ksb/share/jobs/mstatt/151035187731/','init'),[40])    # H
             ,(gm.make_graph('/nfs/slac/g/suncatfs/ksb/share/jobs/mstatt/151036038481/','init'),[40,41]) # NN
             ,(gm.make_graph('/nfs/slac/g/suncatfs/ksb/share/jobs/mstatt/151033093931/','init'),[40,41]) # OH
             ,(gm.make_graph('/scratch/users/ksb/share/jobs/mstatt/151055130379/','init'),[39,40])       # vac NN
           ]
    return graphs[i]


def test_subgraph(p=1):
    from CataLog.structure.graph_utils import make_subgraph,graph_summary,is_subgraph
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

    print is_subgraph(g,g2,ignore_metals=True)
    print is_subgraph(g,g3,ignore_metals=True)
    print is_subgraph(g,g4,ignore_metals=True)

def test_adsorbate(i):
    from CataLog.structure.graph_utils import make_subgraph,graph_summary,is_subgraph,to_atoms
    from ase.visualize import view
    g,ads = adsorbate_graphs(i)
    g2 = make_subgraph(g,ads,1)
    graph_summary(g2)
    view(to_atoms(g))
