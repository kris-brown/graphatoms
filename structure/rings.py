# External Modules
import networkx as nx
import collections,copy,os,itertools
# Internal Modules
import CataLog.structure.graph as GR
from CataLog.misc.utilities import negate
################################################################################

#################
# CLASSES
#--------------

class RingEdge(object):
    """Glorified dictionary - data structure useful for defining water graphs"""
    def __init__(self,fromNode,toNode,dx,dy,dz):
        self.fromNode = fromNode
        self.toNode   = toNode
        self.dx       = dx
        self.dy       = dy
        self.dz       = dz

    def reverse(self):
        return RingEdge(self.toNode,self.fromNode,-self.dx,-self.dy,-self.dz)

    def __str__(self): return '<RingEdge:'+str(self.__dict__)+'>'
    def __eq__(self,other): return self.__dict__ == other.__dict__
    def __repr__(self): return '<Edge:%d %d>'%(self.fromNode,self.toNode)
##############################################################################

class State(object):
    """
    """
    def __init__(self,trajectory,finalNode,remaining_paths,visited_dict):
        self.trajectory      = trajectory # [(Int,(Int,Int,Int))]
        self.finalNode       = finalNode  # Int
        self.remaining_paths = remaining_paths # [RingEdge]
        self.visited_dict = visited_dict        # {(Int,Int,Int) : [RingEdge]}

    def __str__(self): return '<State:'+str(self.__dict__)+'>'
    def __repr__(self):
        d = {k:(v if k!='remaining_paths' else '...') for k,v in self.__dict__.items() }
        return '<State:'+str(d)+'>'

##############################################################################
class DFS(object):
    """Returns all solutions to a search problem with no cost"""
    def __init__(self,succ,actions,is_end):
        #self.init_state = init_state
        self.succ    = succ
        self.actions = actions
        self.is_end  = is_end

        self.verbose=False

    def run_dfs(self,s):
        """
        """
        if self.verbose: print 'entering run_dfs with s = ',s
        new_states = [self.succ(s,a) for a in self.actions(s)]
        results = []

        for ns in new_states:
            if self.verbose: print 'considering new state = ',ns
            end = self.is_end(ns)
            if end:
                if isinstance(end,list):
                    results.append(end)
            else:
                results += self.run_dfs(ns)
        return results

##############################################################################

class RingFinder(object):
    """
    """
    def __init__(self,graph):
        g = graph.to_directed()
        for  i,j,d in g.edges(data=True):
            if i > j: # pbc is correct when i <= j
                for n,d in g[i][j].items():
                    g[i][j][n]['pbc_shift'] = map(negate,g[i][j][n]['pbc_shift'])
        self.graph = g

    def get_rings(self):
        """
        """
        output = []
        h_s     = [n for n,d in self.graph.nodes(data=True) if d['symbol']=='H']
        o_s     = [n for n,d in self.graph.nodes(data=True) if d['symbol']=='O']
        edges  = [RingEdge(i,j,*d['pbc_shift']) for i,j,d in self.graph.edges(data=True)
                        if (i in h_s and j in o_s  )
                        or(i in o_s and j in h_s)]

        o_nodes = [n for n in self.graph.nodes() if self.graph.node[n]['symbol']=='O']
        for i,n in enumerate(o_nodes):
            # print 'finding rings for O node %d (%d/%d)'%(n,i+1,len(o_nodes))
            s = State([(n,(0,0,0))],n,edges,collections.defaultdict(list))
            output.extend(DFS(self.succ,self.actions,self.isEnd).run_dfs(s))

        return output

    def isEnd(self,s):
        """Search is over if no remaining paths OR we've returned home"""
        home = s.trajectory[-1] == (s.finalNode,(0,0,0))
        if home: return s.trajectory # our result
        elif max(map(abs,s.trajectory[-1][1])) > 1:
            #print 'stepped outside of more than 1 pbc'
            return True
        elif len(s.trajectory) > 13:
            #print 'trajectory exceeded 25 steps'
            return True
        elif s.remaining_paths == []: return True
        else: return False

    def succ(self,s,p):
        """Result of stepping along a path from state s"""
        pbc = s.trajectory[-1][1]
        pbc_ = (pbc[0]+p.dx
               ,pbc[1]+p.dy
               ,pbc[2]+p.dz)
        v_dict = copy.deepcopy(s.visited_dict)
        traj   = copy.deepcopy(s.trajectory)
        rem    = [x for x in s.remaining_paths if x!= p]
        v_dict[pbc_].append(p.reverse())
        traj.append((p.toNode,pbc_))

        return State(traj, s.finalNode ,rem,v_dict)

    def actions(self,s):
        """Indices of vailable paths at state s"""
        last,offset = s.trajectory[-1]
        a = filter(lambda p: (p.fromNode == last
                                and p not in s.visited_dict[offset]), s.remaining_paths)
        return a

##############################################################################

def find_water_ring(g):
    """
    """
    if g is None:
        g = GR.GraphMaker(include_frac=0.95).make_graph('/scratch/users/ksb/demos/','water')
    raw_output =  RingFinder(g).get_rings()
    output,seen = [],[]
    for p in raw_output:
        if sorted(p) not in seen:
            output.append(p)
            seen.append(sorted(p))
    return output

def summarize_water_ring(g):
    """
    """
    rings = find_water_ring(g)
    node_output,ring_output = {},collections.defaultdict(int)
    for i in [n for n,d in g.nodes(data=True) if d['symbol']=='O']:
        local_output = collections.defaultdict(int)
        for r in rings:
            len_ring = (len(r)-1)/2
            ring_output[len_ring]+=1
            if r[0][0]==i:
                local_output[len_ring]+=1
        node_output[i] = dict(local_output)
    return ring_output
"""
def count_H_bonds(g,min_bo=0.04,max_bo=0.2):
    Count number of hydrogen bonds
    hbonds = 0
    for i,j,d in g.edges(data=True):
        if min_bo < d['bondorder'] < max_bo: # DISTANCE CHECK TO MAKE SURE NOT COVALENT BOND
            if ((g.node[i]['symbol'],g.node[j]['symbol'])) in [('H','O'),('O','H')]: hbonds+=1
    return hbonds
"""
def count_bonds(g,elems,min_bo=0,max_bo=4):
    """
    Count all bonds between some set of elements, subject to some bondorder range
    """
    bonds =  0
    pairs = list(itertools.permutations(elems))
    for i,j,d in g.edges(data=True):
        if min_bo < d['bondorder'] < max_bo: # DISTANCE CHECK TO MAKE SURE NOT COVALENT BOND
            if ((g.node[i]['symbol'],g.node[j]['symbol'])) in pairs: bonds+=1
    return bonds

# Count Hydrogen Bonds
def count_H_bonds(g): return count_bonds(g,['H','O'],min_bo=0.04,max_bo=0.2)

#########
# Testing
#########
def water_graphs():
    """
    """
    water_root = '/scratch/users/ksb/demos/tom_demo/percentiles/'
    gs = []
    for d in sorted(os.listdir(water_root)):
        print 'Precessing ',d
        g = GR.GraphMaker(min_bo=0.05
                      ,include_frac=0.95).make_graph(water_root+d,'water_slab')
        print summarize_water_ring(g)
        hbonds = 0
        for i,j,d in g.edges(data=True):
            if d['weight'] > 1.3:
                if ((g.node[i]['symbol'],g.node[j]['symbol'])) in [('H','O'),('O','H')]: hbonds+=1
        print 'Hbonds are ',hbonds


"""
Old function which does not account for periodicity
Maybe still relevant in some circumstances?
def water_rings(stordir,trajname,include_frac=0.95):
    gm = GR.GraphMaker(include_frac=include_frac)
    G = gm.make_graph(stordir,trajname)
    a = gm.make_atoms(stordir,trajname)
    remove = [node for node in G.nodes() if a[node].symbol not in ['H','O']]
    G.remove_nodes_from(remove)
    cycles,seen = [],[]
    print 'Cycles are:'
    for c in nx.simple_cycles(G.to_directed()):
        if len(c) > 2 and sorted(c) not in seen:
            cycles.append(c)
            seen.append(sorted(c))
            print c

    return cycles
"""
