# External Modules
from typing      import  Tuple,List,Dict
from os.path     import join
from collections import defaultdict
from json        import dump
from ase         import Atoms,Atom  # type: ignore
import numpy as np                  # type: ignore

# Internal Modules
from dbgen.core.lists           import flatten

################################################################################

def parse_chargemol_sherlock(stordir : str,atoms : Atoms) -> None:
    """
    Analyzes the contents of a chargemol calculation directory.

    Returns an unzipped list of <ind,ind,bondorder> triples
    followed by an unzipped list of <ind,charge> pairs

    """

    # Auxillary Functions
    #--------------------
    def parse_line(line : str) -> Tuple[int,float,List[int]]:
        """
        Get bonded atom, bond order, and offset
        """
        assert line[:16]==' Bonded to the (', "No parse line ->"+line+'<- %d'%len(line)
        offStr,indStr,boStr = line[16:].split(')')
        offset = [int(x) for x in offStr.split(',')] # Chunk containing offset info
        ind    = int(indStr.split()[-3]) - 1        # Chunk containing index info (chargemol starts at 1, not 0)
        bo     = float(boStr.split()[4])            # Chunk containing B.O. info
        return (ind,bo,offset)

    def mk_pbc_dict(atoms : Atoms
                   ) -> Dict[int,Tuple[int,int,int]]:
        """
        Helpful docstring
        """
        def g(tup : tuple)->Tuple[int,int,int]:
            """
            Helper function to yield tuples for pbc_dict
            """
            def f(x : float)->int:
                """
                Helper function for g
                """
                if   x < 0: return -1
                elif x < 1: return 0
                else:       return 1

            x,y,z = tup
            return (f(x),f(y),f(z))

        scaled_pos  = atoms.get_scaled_positions(wrap=False).tolist()
        scaled_pos_ = zip(range(len(atoms)),scaled_pos)
        pbc_dict    = {i : g(p) for i,p in scaled_pos_}
        return pbc_dict

    def dict_diff(d1 : Dict[int,np.array]
                 ,d2 : Dict[int,np.array]
                 ) -> Dict[int,np.array]:
        """
        Helpful docstring
        """
        return {i: np.array(d2[i]) - np.array(d1[i]) for i in d1.keys()}


    # Handle sortdict
    #----------------
    try:
        with open(join(stordir,'ase-sort.dat'),'r') as f:
            loglines = map(lambda x: map(int,x.split()),f.readlines())
        sortdict = {x:y for x,y in loglines} # type: Dict[int,int]
    except IOError:
        sortdict = {}

    # Classes
    #--------

    class PotentialEdge(object):
        """
        Container for information we need to decide later if it's a
        graph-worthy bond. This class is nothing more than a dictionary.
        """
        def __init__(self
                    ,fromNode   : int
                    ,toNode     : int
                    ,bond_id    : int
                    ,distance   : float
                    ,offset     : List[int]
                    ,bond_order : float
                    ) -> None:

            self.fromNode = fromNode
            self.toNode   = toNode
            self.bond_id  = bond_id
            self.pbc_x    = offset[0]
            self.pbc_y    = offset[1]
            self.pbc_z    = offset[2]
            self.distance = distance
            self.bondorder= bond_order


    class BondOrderSection(object):
        """
        Process one section of the Bond Order output of Chargemol
        """
        def __init__(self
                    ,ind       : int
                    ,raw_lines : List[str]
                    ,pbcdict   : dict
                    ) -> None:

            self.ind      = ind
            self.bonds    = [parse_line(x) for x in raw_lines]
            self.pbcdict  = pbcdict

            self.bondcounter = defaultdict(int) # type: Dict[Tuple[int,int],int]

        def _relative_shift(self
                           ,i : int
                           ,j : int
                           ) -> np.array:
            """
            Given a pbc_dict and two indices, return the original pbc shift
            for a bond from i to j
            """
            pi,pj = [np.array(self.pbcdict[x]) for x in [i,j]]
            return pj - pi

        def makeEdge(self
                    ,tup :  Tuple[int,float,List[int]]
                    ) -> PotentialEdge:
            """
            Creates an Edge instance from the result of a parsed Bond Order log line
            """
            (toInd,bo,offset) = tup
            fromInd = self.ind

            if sortdict: # undo VASP reordering if necessary
                fromInd,toInd = [sortdict[x] for x in [fromInd,toInd]] # type: ignore

            # correct for WRAPPING atoms
            offset = (np.array(offset) +  self._relative_shift(fromInd,toInd)).tolist()

            shift = np.dot(offset, atoms.get_cell()) # PBC shift
            p1    = atoms[fromInd].position
            p2    = atoms[toInd].position + shift
            d     = np.linalg.norm(p2-p1)

            bondid = self.bondcounter[(fromInd,toInd)]
            self.bondcounter[(fromInd,toInd)] += 1
            return PotentialEdge(fromInd,toInd,bondid,d,offset,bo)

        def make_edges(self)->List[PotentialEdge]:
            """Apply edgemaker to result of parsing logfile lines"""
            return  [self.makeEdge(b) for b in self.bonds]

    def parse_chargemol_pbc(header_lines : List[str]
                           ,cell         : List[float]
                           ) -> Dict[int,Tuple[int,int,int]]:
        """
        Helpful docstring
        """
        atoms = Atoms(cell=cell)

        for i,l in enumerate(header_lines[2:]):
            try:
                if '*' in l:
                    l = l.replace('*','') + ' ???' #Asterisks are touching the z value
                s,x,y,z,_ = l.split()
                p = [float(q) for q in [x,y,z]]
                atoms.append(Atom(s,position=p))
            except Exception as e:
                pass #print('exception???',e)

        return mk_pbc_dict(atoms)

    # Main Program
    #-------------
    header,content    = [], [] # type: Tuple[List[str],List[str]]
    sections          = []     # type: List[BondOrderSection]
    filtedges         = []
    head_flag,counter = True,-1       # Initialize
    filepath  = join(stordir,'DDEC6_even_tempered_bond_orders.xyz')# File to parse
    pbcdict   = mk_pbc_dict(atoms) # the 'true' PBC coordinate of each atom

    for line in open(filepath,'r'):
        if not (line.isspace() or line[:4] == ' 201'): # remove blank lines and calendar date
            if line[1]=='=':
                head_flag = False                                            # no longer in header  section
                chargemol_pbc = parse_chargemol_pbc(header,atoms.get_cell()) # PBC of chargemol atoms

            elif head_flag: header.append(line)     # we're still in the header section
            elif 'Printing' in line:                # we're describing a new atom
                content = []                        # reset buffer content
                counter += 1                        # update index of our from_index atom
            elif 'sum' in line:                     # summary line at end of a section
                dic = dict_diff(pbcdict,chargemol_pbc)
                edgs = BondOrderSection(counter,content,dic).make_edges()
                filtedges.extend([e for e in edgs
                    if  (e.bondorder > 0.001) and (e.fromNode <= e.toNode)])
            else:
                content.append(line)              # business-as-usual, add to buffer

    with open('bonds.json','w') as f:
        dump([e.__dict__ for e in filtedges],f)

    if False:
        #############
        # Get charges
        #------------

        fpath     = join(stordir,'DDEC6_even_tempered_net_atomic_charges.xyz')
        with open(fpath,'r') as f:
            chg_lines = f.readlines()

        head_flag = True
        tail_flag = False
        chgs      = [] # List[float]

        for line in chg_lines[:-3]:
            if 'eigenvalues' in line:
                head_flag = False               # we are no longer in the header region
            if not head_flag and not tail_flag:
                if line.isspace():
                    tail_flag = True            # we are now in the tail region
                else:
                    try: # we are in the region with charges, for the most part
                        chgs.append(float(line.split()[5])) # first line bogus
                    except ValueError:
                        pass
        inds = list(range(len(atoms)))

        if sortdict: # undo VASP reordering if necessary
            charges = [chgs[sortdict[i]] for i in inds] # type: ignore
        else:
            charges = chgs

        with open('charges.json','w') as f:
            dump(enumerate(charges),f)


if __name__=='__main__':
    from sys    import argv
    from glob   import glob
    from ase.io import read # type: ignore
    pth   = argv[1]
    trajs = glob(join(pth,'*.traj'))
    traj  = read(trajs[0])
    parse_chargemol_sherlock(pth,traj)
