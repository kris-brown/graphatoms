# External Modules
import ase #type: ignore

# Internal Modules
from graphatoms.misc.atoms import angle,dihedral #type: ignore



##############################################################################

def nearest_neighbor(atoms,atom,exclude=[],include=None):
    """Get closest atom (ignoring PBC), subject to include/exclude constraints"""
    exclude.append(atom.index) #don't include itself in the search
    if include==None: include=range(len(atoms))
    apos = atom.position
    minD = 100
    nn   = None
    for a in atoms:
        if a.index not in exclude and a.index in include:
            if np.sum(np.square(a.position-apos)) < minD:
                nn = a
    return nn

def center(atoms,ind):
    """Centers x,y coordinates about a particular atom"""
    a = atoms.copy()
    p = a.get_positions()[ind]
    c = a.get_cell()
    a.positions -= [p[0],p[1],0] - (c[0]+c[1])/2
    a.wrap(pbc=[1,1,0])
    return a


def atomsZ(atoms):
    """
    Converts atoms object to a 'normal form' Z Matrix that is insensitive to noise
    Ideally it will make a list of Z matrix when slight deviations could
    result in different structures

    Some security is granted by making the 'highest force' atom the starting point
        and centering the x,y coordinates about this atom

    NEED A canonical orientation for the unit cell
    before saving cell, need to rotate it so that the X vector is aligned
    with the first two atoms, for example
    """
    import emt #type: ignore
    atoms.set_calculator(emt.EMT())
    atoms.get_potential_energy()
    forces  = atoms.get_forces()
    print('Calculated EMT forces')
    srted   = sorted([(np.linalg.norm(f),a) for f,a in zip(forces,atoms)])
    ordered = [a for f,a in reversed(srted)] # list of atoms, decreasing forces
    centered = center(atoms,ordered[0].index)
    print('Centered about high-force atom')
    n    = len(atoms)
    c    = atoms.get_cell()
    a0   = centered[0]
    r0   = ZRow([],[a0.symbol])
    if n == 1: return ZMatrix(c,[r0])
    a1   = nearest_neighbor(centered,a0)
    print('got nearest neighbor')
    v01  = a1.position-a0.position
    r1   = ZRow([r0],[a1.symbol,round(np.linalg.norm(v01),2)])
    if n == 2: return ZMatrix(c,[r0,r1])
    a2   = nearest_neighbor(centered,a1,[a0])
    v12  = a2.position-a1.position
    a012 = round(angle(v01,v12),2)
    r2   = ZRow([r0,r1],[a2.symbol,round(np.linalg.norm(v12),2),a012])
    if n == 3: return ZMatrix(c,[r0,r1,r2])
    rs   = get_remaining_rows(centered,{a0.index:r0,a1.index:r1,a2.index:r2})
    return ZMatrix(c[r0,r1,r2]+rs)

def get_remaining_rows(atoms,existing_row_dict):
    """Recursive function which returns a list of Z matrix rows, given a partial
       list of Z matrix rows"""
    if len(existing_row_dict)==len(atoms): return []
    existing_inds  = existing_row_dict.keys()
    existing_atoms = [a for a in atoms if a.index in existing_inds]
    center_of_existing = get_center(existing_atoms)
    new_atom = nearest_neighbor(atoms,center_of_existing,exclude=existing_inds)

    atom_x = nearest_neighbor(atoms,new_atom,include=existing_inds)
    atom_y = nearest_neighbor(atoms,atom_x,include=existing_inds)
    atom_z = nearest_neighbor(atoms,atom_x,exclude=[atom_x.ind],include=existing_inds)

    vx   = atom_x.position - new_atom.position
    vxy  = atom_y.position - atom_x.position # THESE NEED TO BE DONE IN A WAY THAT PREVENTS > 180 angles?
    vyz  = atom_z.position - atom_y.position

    bx   = np.linalg.norm(vx)
    axy  = angle(vx,vxy)
    dxyz = dihedral(vx,vxy,vyz)
    rx   = existing_row_dict[atom_x.index]
    ry   = existing_row_dict[atom_y.index]
    rz   = existing_row_dict[atom_z.index]
    new_row = ZRow([rx,ry,rz],[new_atom.symbol,bx,axy,dxyz])

    existing_row_dict[new_atom.index]=new_row
    return new_row + get_remaining_rows(atoms,existing_row_dict)

def get_center(atoms):
    """returns position of central-most atom in a list of atoms"""
    com    = atoms.center()
    minD   = 100
    for a in atoms:
        if np.sum(np.square(a.position-com)) < minD:
            cent = a
    return cent

class ZRow(object):
    def __init__(self,zrows,vals):
        self.vals=vals   #list of 1-4 properties (symbol,bond,angle,dihedral)
        self.zrows=zrows #list of 0-3 other ZRow objects which the vals refer to
    def __str__(self): return str(self.vals)

class ZMatrix(object):
    def __init__(self,cell,zrows):
            self.cell=cell
            self.zrows=zrows
            self.n = len(zrows)
    def __str__(self):
        zDict = {r:i for i,r in enumerate(self.zrows) if i > 2}
        s     = "R0: {0}".format(self.zrows[0].vals[0])
        if self.n == 1: return s
        s+= "\nR1: {0} (R0) {1}".format(*self.zrows[1].vals)
        if self.n == 2: return s
        s+= "\nR2: {0} (R1) {1} (R0) {2}".format(*self.zrows[2].vals)
        for r in zDict:
            rrows = [zDict[r]] + map(zDict.get,r.zrows)
            args  = zip(rrows,r.vals)
            s+="\nR{0}: {1} (R{2}) {3} (R{4}) {5} (R{6}) {7}".format(*args)
        return s


    def to_atoms(self):
        """create Atoms object"""
        raise NotImplementedError

def main():
    testAtoms = ase.Atoms(numbers=[1,1],positions=[[10,10,10],[10,10,11]],cell=[[20,0,0],[0,20,0],[0,0,20]])
    #testAtoms2 = ase.Atoms([1,1],[[10,10,10.01],[10,10,11]],[[20,0,0],[0,20,0],[0,0,20]])
    print(atomsZ(testAtoms))
