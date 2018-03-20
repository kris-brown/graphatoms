# External Modules
from typing import List,Tuple,Dict,Callable
import networkx as nx #type: ignore
import collections,copy,os,itertools

# Internal Modules
import CataLog.structure.graph as GR #type: ignore
from CataLog.misc.utilities import negate,DFS #type: ignore
################################################################################

TrajectoryStep = Tuple[int,Tuple[int,int,int]]
Trajectory     = List[TrajectoryStep]
#########
# CLASSES
#--------

class RingEdge(object):
    """
    Glorified dictionary - data structure useful for defining water graphs
    """
    def __init__(self
                ,fromNode : int
                ,toNode   : int
                ,dx       : int
                ,dy       : int
                ,dz       : int
                ) -> None:
        self.fromNode = fromNode
        self.toNode   = toNode
        self.dx       = dx
        self.dy       = dy
        self.dz       = dz

    def reverse(self) -> RingEdge:
        return RingEdge(self.toNode,self.fromNode,-self.dx,-self.dy,-self.dz)

    def __str__(self) -> str:
        return '<RingEdge:'+str(self.__dict__)+'>'

    def __eq__(self,other) -> bool:
        return self.__dict__ == other.__dict__
        
    def __repr__(self)->str:
        return '<Edge:%d %d>'%(self.fromNode,self.toNode)
##############################################################################

class State(object):
    """
    helpful docstring
    """
    def __init__(self
                ,trajectory : Trajectory
                ,finalNode  : int
                ,remaining_paths : List[RingEdge]
                ,visited_dict    : Dict[Tuple[int,int,int],List[RingEdge]]
                ) -> None:
        self.trajectory      = trajectory
        self.finalNode       = finalNode
        self.remaining_paths = remaining_paths
        self.visited_dict = visited_dict        # {(Int,Int,Int) : [RingEdge]}

    def __str__(self) -> str:
        return '<State:'+str(self.__dict__)+'>'

    def __repr__(self) -> str:
        d = {k:(v if k!='remaining_paths' else '...') for k,v in self.__dict__.items() }
        return '<State:'+str(d)+'>'

##############################################################################
##############################################################################
def result(s : State) -> Trajectory:
    """
    Returns search trajectory iff we arrived back home
    """
    home = s.trajectory[-1] == (s.finalNode,(0,0,0))
    if home:
        return s.trajectory # our result
    else:
        return None

def isEnd(s : State) -> bool:
    """
    Search is over if no remaining paths OR we've returned home
    """
    laststep = s.trajectory[-1]
    home     = laststep == (s.finalNode,(0,0,0))
    x,y,z    = laststep[1] # pbc's of last step
    ax,ay,az = abs(x),abs(y),abs(z)

    if home: return True
    elif max([ax,ay,az]) > 1:
        return True # stepped outside of more than 1 pbc'
    elif len(s.trajectory) > 13:
        return True # exceeded max number of steps
    elif s.remaining_paths == []:
        return True # no more paths
    else: return False

def succ(s : State
        ,p : RingEdge
        ) -> State:
    """
    Result of stepping along a path from state s
    """
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

def actions(s : State)->List[RingEdge]:
    """
    Indices of vailable paths at state s
    """
    last,offset = s.trajectory[-1]
    def f(p): return (p.fromNode == last
                  and p not in s.visited_dict[offset])
    a = filter(f, s.remaining_paths)
    return a


##############################################################################

def preprocess_graph(graph : nx.Graph) ->  nx.Graph:
    """
    Helpful docstring
    """
    g = graph.to_directed()
    for i,j,d in g.edges(data=True):
        if i > j: # pbc is correct when i <= j
            for n,d in g[i][j].items():
                g[i][j][n]['pbc_shift'] = map(negate,g[i][j][n]['pbc_shift'])
    return g

def get_rings(graph_in : nx.Graph)->List[Trajectory]:
    """
    Helpful docstring
    """
    graph = preprocess_graph(graph_in)
    output = [] # type: list
    h_s     = [n for n,d in graph.nodes(data=True) if d['symbol']=='H']
    o_s     = [n for n,d in graph.nodes(data=True) if d['symbol']=='O']
    edges  = [RingEdge(i,j,*d['pbc_shift']) for i,j,d in graph.edges(data=True)
                    if (i in h_s and j in o_s  )
                    or(i in o_s and j in h_s)]

    o_nodes = [n for n in graph.nodes() if graph.node[n]['symbol']=='O']
    for i,n in enumerate(o_nodes):
        # print 'finding rings for O node %d (%d/%d)'%(n,i+1,len(o_nodes))
        s = State([(n,(0,0,0))],n,edges,collections.defaultdict(list))
        output.extend(DFS(succ,actions,isEnd,result).run_dfs(s))

    return output



def find_water_ring(g : nx.Graph
                   )->List[Trajectory]:
    """
    Helpful docstring
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

def summarize_water_ring(g : nx.Graph) -> dict:
    """
    Helpful docstring
    """
    rings = find_water_ring(g)
    node_output,ring_output = {},collections.defaultdict(int) #type: dict,dict
    for i in [n for n,d in g.nodes(data=True) if d['symbol']=='O']:
        local_output = collections.defaultdict(int) # type: dict
        for r in rings:
            len_ring = (len(r)-1)/2
            ring_output[len_ring]+=1
            if r[0][0]==i:
                local_output[len_ring]+=1
        node_output[i] = dict(local_output)
    return ring_output

def count_bonds(g       : nx.Graph
               ,elems   : List[str]
               ,min_bo  : float     = 0
               ,max_bo  : float     = 4
               ) -> int:
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
def count_H_bonds(g : nx.Graph)-> int:
    return count_bonds(g,['H','O'],min_bo=0.04,max_bo=0.2)

#########
# Testing
#########
def water_graphs() -> None:
    """
    Demo
    """
    water_root = '/scratch/users/ksb/demos/tom_demo/percentiles/'
    gs = [] # type: list
    for d in sorted(os.listdir(water_root)):
        print('Precessing ',d)
        gi = GR.GraphInput(water_root+d,'water_slab')
        g = GR.GraphMaker(min_bo=0.05,include_frac=0.95).make_graph(gi)
        print(summarize_water_ring(g))
        hbonds = 0
        for i,j,data in g.edges(data=True):
            if data.get('weight') > 1.3:
                if ((g.node[i]['symbol'],g.node[j]['symbol'])) in [('H','O'),('O','H')]: hbonds+=1
        print('Hbonds are ',hbonds)
