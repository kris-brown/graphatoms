#External Modules
import ase,json,math
import numpy as np
################################################################################
"""
Functions for interacting with ASE

Element Lists
    nonmetal_symbs
    nonmetals
    mag_elems
    mag_nums
    paw_elems
    symbols2electrons

Atoms object manipulation
    classify_system
    make_atoms
    cell_to_param
    restore_magmom

Geometry
    angle
    dihedral

Site Related
    get_sites

Keld Data related
    get_keld_data
    get_expt_surf_energy

Antiquated
    inversion_center
    make_cell_positive
    construct_atoms

"""

# Keld Data Related
#-----------------
def get_keld_data(name,k):
    import data_solids_wPBE

    name = name.split('_')[0] #sometimes suffix ('bulkmod') is appended to name
    try: return data_solids_wPBE.data[data_solids_wPBE.getKey[name] ][k]
    except KeyError:
        try: return data_solids_wPBE.data[data_solids_wPBE.getKey[name] ][k+' kittel'] #for 'bulkmodulus' / 'bulkmodulus kittel'
        except KeyError: return None

def get_expt_surf_energy(name):
	import data_sol53_54_57_58_bm32_se30 as d
	if 'x' not in name or '_' not in name: return None

	metal,crystal = name.split('_')[0].split('-')
	facet = name.split('_')[1].replace(',','')
	if crystal=='hcp' and facet == '001': facet = '0001'
 	try: return d.get_exp_surface_energy(metal+facet)[1]*0.0624 #convert J/m^2 to eV/A^2
	except KeyError: return None


# Atoms object manipulation
def restore_magmom(trajpckl):
    traj = pickle.loads(trajpckl)
    try:
        mags = traj.get_magmoms()
        if any([x>0 for x in mags]): traj.set_initial_magnetic_moments([3 if e in magElems else 0 for e in traj.get_chemical_symbols()])
    except: pass
    return pickle.dumps(traj)


def cell_to_param(cell):
    """ANGLES ARE IN RADIANS"""
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    alpha = angle(cell[1],cell[2])
    beta  = angle(cell[0],cell[2])
    gamma = angle(cell[0],cell[1])
    return (a,b,c,alpha,beta,gamma)
def classify_system(atoms):
    cutoff    = 6  # A
    atoms.center() # In case dealing with Nerds Rope & Co.
    minx,miny,minz,maxx,maxy,maxz = 1000,1000,1000,-1000,-1000,-1000
    for a in atoms:
        if a.position[0] < minx: minx =  a.position[0]
        if a.position[1] < miny: miny =  a.position[1]
        if a.position[2] < minz: minz =  a.position[2]
        if a.position[0] > maxx: maxx =  a.position[0]
        if a.position[1] > maxy: maxy =  a.position[1]
        if a.position[2] > maxz: maxz =  a.position[2]

    cell_abc  = map(np.linalg.norm,atoms.get_cell()[:3])
    thickness = [maxx-minx,maxy-miny,maxz-minz]
    pbc       = [cell_abc[i]-thickness[i] < cutoff for i in range(3)]

    if   all(pbc): return 'bulk'
    elif any(pbc): return 'surface'
    else:          return 'molecule'

def make_atoms(*args):
    from ase import Atoms
    from ase.constraints import FixAtoms
    import numpy as np
    n,p_t,c,m,cs = map(json.loads,args)
    return Atoms(numbers=n,positions=zip(*p_t)
            ,cell=c,magmoms=m
            ,constraint=FixAtoms(np.nonzero(cs)[0]))



# Geometry

def angle(v1,v2):
    return np.arccos(np.dot(v1,np.transpose(v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)))
def dihedral(v1,v2,v3):
    """http://azevedolab.net/resources/dihedral_angle.pdf"""
    x12 = np.cross(v1,v2)
    x23 = np.cross(v2,v3)
    n1  = x12/np.linalg.norm(x12)
    n2  = x23/np.linalg.norm(x23)
    u1  = n2
    u3  = v2/np.linalg.norm(v2)
    u2  = np.cross(u3,u1)
    cos_theta = np.dot(n1,u1)
    sin_theta = np.dot(n1,u2)
    theta = -math.atan2(sin_theta,cos_theta)
    return theta


##############
# Element lists
###############
nonmetal_symbs   = ['H','C','N','P','O','S','Se','F','Cl','Br','I','At','He','Ne','Ar','Kr','Xe','Rn']
nonmetals        = [ase.data.chemical_symbols.index(x) for x in nonmetal_symbs]

relevant_atoms = [1,3,4,6,7,8,9,11,12,13,14,16,17] \
                    + range(19,36)+range(37,52)+[55,56]+range(72,80)

relevant_nonmetals = list(set(nonmetals) & set(relevant_atoms))

mag_elems        = ['Fe','Mn','Cr','Co','Ni']
mag_nums         = [24,25,26,27,28]
paw_elems        = ['Li','Be','Na','Mg','K','Ca','Rb','Sr','Cs','Ba','Zn']

def symbols2electrons(symbols,psp='gbrv'): # added Be, Cd on my own
    symbdict = {'gbrv':{'Ag':19,'Al':3,'As':5,'Au':11,'Ba':10,'Br':7,'B':3,'Be':4,'Ca':10,'Cd':12,'Co':17,'Cr':14,'Cs':9,'C':4,'Cu':19,'Fe':16,'F':7,'Ga':19,'Ge':14,'Hf':12,'Hg':12,'H':1,'In':13,'Ir':15,'I':7,'K':9,'La':11,'Li':3,'Mg':10,'Mn':15,'Mo':14,'Na':9,'Nb':13,'Ni':18,'N':5,'Os':16,'O':6,'Pb':14,'Pd':16,'Pt':16,'P':5,'Rb':9,'Re':15,'Rh':15,'Ru':16,'Sb':15,'Sc':11,'Se':6,'Si':4,'Sn':14,'Sr':10,'S':6,'Ta':13,'Tc':15,'Te':6,'Ti':2,'Tl':13,'V':13,'W':14,'Y':11,'Zn':20,'Zr':12}}
    return sum([symbdict[psp][x] for x in symbols])

#############
# Site Related
###############
def make_pmg_slab(a,facet):
    from pymatgen.io.ase   import AseAtomsAdaptor
    from pymatgen.core.surface  import SlabGenerator,Slab
    lattice             = a.get_cell()
    species             = a.get_chemical_symbols()
    coords              = a.get_positions()
    miller_index        = facet
    oriented_unit_cell  = AseAtomsAdaptor.get_structure(a)
    shift               = 0    #???????
    scale_factor        = None #???????
    return Slab(lattice, species, coords, miller_index,oriented_unit_cell, shift, scale_factor,coords_are_cartesian=True)

def get_sites(a,facet, site_type = 'all',symm_reduce = 0.01,height = 0.9):
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    assert site_type in ['all','bridge','ontop','hollow'], 'Please supply a valid site_type'

    slab = make_pmg_slab(a,facet)
    sites  = AdsorbateSiteFinder(slab,height = height).find_adsorption_sites(symm_reduce=symm_reduce, distance = 0)[site_type]
    return sites

def show_sites(a,facet,site_type = 'all',symm_reduce = 0.01, height = 0.9):
    from misc.print_parse import plot_slab
    import matplotlib.pyplot as plt
    slab = make_pmg_slab(a,facet)
    plot_slab(slab,plt.gca(),repeat=3,site_type = site_type, symm_reduce=symm_reduce, height = height)
    plt.show() # user looks, closes plot to continue

def get_mic_distance(p1, p2, cell, pbc,dis_ind = 'xyz'):
    """ This method calculates the shortest distance between p1 and p2
         through the cell boundaries defined by cell and pbc.
         This method works for reasonable unit cells, but not for extremely
         elongated ones.
    """
    import itertools
    ct = cell.T
    pos = np.mat((p1, p2))
    scaled = np.linalg.solve(ct, pos.T).T
    for i in range(3):
        if pbc[i]:
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    P = np.dot(scaled, cell)

    pbc_directions = [[-1, 1] * int(direction) + [0] for direction in pbc]
    translations = np.mat(list(itertools.product(*pbc_directions))).T
    p0r = np.tile(np.reshape(P[0, :], (3, 1)), (1, translations.shape[1]))
    p1r = np.tile(np.reshape(P[1, :], (3, 1)), (1, translations.shape[1]))
    dp_vec = p0r + ct * translations
    if dis_ind == 'xyz':
        squared_dis = np.power((p1r - dp_vec), 2).sum(axis=0)
    elif dis_ind =='xy':
        squared_dis = np.power((p1r - dp_vec)[0:2], 2).sum(axis=0)
    else:
        raise ValueError, 'Please provide valid direction to include in distance \'xy\' or \'xyz\''
    d = np.min(squared_dis)**0.5
    return d

def get_mic_vector(p1, p2, cell, pbc,dis_ind = 'xyz'):
    """ This method calculates the shortest distance between p1 and p2
         through the cell boundaries defined by cell and pbc.
         This method works for reasonable unit cells, but not for extremely
         elongated ones.
    """
    import itertools
    ct = cell.T
    pos = np.mat((p1, p2))
    scaled = np.linalg.solve(ct, pos.T).T
    for i in range(3):
        if pbc[i]:
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    P = np.dot(scaled, cell)
    import pdb
    pdb.set_trace()
    pbc_directions = [[-1, 1] * int(direction) + [0] for direction in pbc]
    translations = np.mat(list(itertools.product(*pbc_directions))).T
    p0r = np.tile(np.reshape(P[0, :], (3, 1)), (1, translations.shape[1]))
    p1r = np.tile(np.reshape(P[1, :], (3, 1)), (1, translations.shape[1]))
    dp_vec = p0r + ct * translations
    if dis_ind == 'xyz':
        squared_dis = np.power((p1r - dp_vec), 2).sum(axis=0)
    elif dis_ind =='xy':
        squared_dis = np.power((p1r - dp_vec)[0:2], 2).sum(axis=0)
    else:
        raise ValueError, 'Please provide valid direction to include in distance \'xy\' or \'xyz\''
    d = np.min(squared_dis)**0.5
    return d


def delete_atom(atoms_obj, ind):
    new_atoms_obj = atoms_obj.copy()
    del new_atoms_obj[ind]
    return new_atoms_obj

def delete_vacancy(parent_atoms_obj, child_atoms_obj):
    vac_pos, vac_ind = get_vacancy_pos(parent_atoms_obj, child_atoms_obj)
    return delete_atom(parent_atoms_obj,vac_ind)

def delete_adsorbates(atoms_obj, adsorbates_list):
    from CataLog.misc.utilities import flatten
    new_atoms_obj = atoms_obj.copy()
    reverse_ordered_adsorbate_list = np.sort(flatten(adsorbates_list))[::-1]
    for ind in reverse_ordered_adsorbate_list:
        print ind
        new_atoms_obj = delete_atom(new_atoms_obj, ind)
    return new_atoms_obj

def did_it_restructure(atoms_1, atoms_2, restructure_criteria = 1):
    distances_1 = np.linalg.norm(atoms_1.positions, axis = 1)
    distances_2 = np.linalg.norm(atoms_2.positions, axis = 1)
    max_dis = np.max(np.abs(distances_2-distances_1))
    vert_dis = np.max(atoms_1.positions[:,2] - atoms_2.positions[:,2])
    return np.max(np.abs(distances_2-distances_1))>restructure_criteria

def get_vacancy_pos(parent, child):
    """Returns the list of length three indicating the location of the vacancy
    position on the slab using the bare_slab parent indicated in the database.
    The parent should be the atoms object of the relaxed bare slab with no"""
    import numpy as np
    if len(parent) < len(child):
        return None
    #Find the distances between each atom in the parent object and each of the atoms
    #in the child object
    #returns list with a length equal to that of the parents object
    distances = map(lambda pos: map(np.linalg.norm,(pos-child.positions)), parent.positions)
    #Find the nearest neighbor between the parent atom and the child atoms
    min_dis = map(np.min, distances)
    #the atom with the largest distance between the parent and child atoms is assumed to be
    # at the location of the vacancy
    return parent.positions[np.argmax(min_dis)], np.argmax(min_dis)

def view_asv_list(asv_list):
    from ase.visualize import view
    atoms_list = map(lambda x: x.get_adsorbed_surface(), asv_list)
    formulas_list = map(lambda x: x.get_chemical_formula(), atoms_list)
    num_diff_forms = len(list(set(formulas_list)))
    array_of_atoms_arrays = {formula_curr : [] for formula_curr in list(set(formulas_list))}
    for atoms_curr, formula_curr in zip(atoms_list, formulas_list):
        array_of_atoms_arrays[formula_curr].append(atoms_curr)
    [view(atoms_curr) for atoms_curr in array_of_atoms_arrays.values()]

def match_indices(atoms1,atoms2):
    """
    Returns mapping dictionary that maps indices in atoms2 to be as 'close'
    to atoms1 as possible
    """
    from scipy.spatial.distance import cdist

    rang  = range(len(atoms1))
    a1,a2 = atoms1.copy(),atoms2.copy()                               # Copy these guys b/c we're gonna center()
    c1,c2 = [a.constraints[0].get_indices() for a in [atoms1,atoms2]] # Get lists of constrained atoms

    a1.center(),a2.center()                  # Remove translation as a degree of freedom
    dists = cdist(a1.positions,a2.positions) # Pairwise distance matrix

    for i in rang:                                 # For every pair
        for j in rang:                             # If we have
            if (atoms1[i].symbol!=atoms2[j].symbol # Different elements or
                     or ((i in c1) != (j in c2))): # One is constrained but not the other, then
                dists[i,j] = 10000                 # Inflate distance; CLEARLY not correct atom

    out = {}  # Initialize our mapping dictionary

    for i in rang:                         # For every atom in atoms object 1...
        newind = np.argmin(dists[i,:])     # Find the closest atom in 2 to the current atom in 1
        out[i] = newind                    # Insert old atom into new Atoms object
        dists[:,newind] = 10000            # We don't ever want to select this again

    return out

def test_match():
    """
    Demonstrates the match_indices function on a 1x1x6 slab with its bottom atom
    moved to the end of the index list
    """
    from ase.io import read
    a1 = read('/scratch/users/ksb/fireworks/jobs/relax_2017_11_04_11_05_10/final.traj')
    a2 = read('/scratch/users/ksb/fireworks/jobs/relax_2017_11_04_11_05_10/init.traj')
    at1 = a2.pop(0) # make sure switch respects constraints!
    a2.append(at1)  # elements of a2 are now reordered
    print 'a1 (constrained = %s)'%(a1.constraints[0])
    for a in a1: print a
    print '\na2 (constrained = %s)'%(a2.constraints[0])
    for a in a2: print a
    print 'result: ',match_indices(a1,a2)
    # should be {0 -> 5, 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4}


###########
# ANTIQUATED
#############
"""

def inversion_center(atoms,tol= 0.01):
	#Returns whether or not Atoms object has an inversion center
	centeredPos = atoms.get_positions() - atoms.get_center_of_mass()

	while len(centeredPos) > 0:
		p1, centeredPos = centeredPos[0], centeredPos[1:]
		found = False
		for i,p2 in enumerate(centeredPos):
			if np.linalg.norm(p1+p2) < tol:
				centeredPos = np.delete(centeredPos,i,0)
				found=True
				break
		if not found: 	return False
	return True


def make_cell_positive(atoms):
    #Flips atoms objects to have positive positive cell directions
    atoms_new = atoms.copy()
    new_pos = atoms.get_positions()
    for direction in [0,1,2]:
        c = atoms.cell[direction]/np.linalg.norm(atoms.cell[direction])
        if sum(c)<0:
            new_pos = [x-atoms.cell[direction] for x in new_pos]
            atoms_new.cell[direction] *= -1
    atoms_new.positions = new_pos
    return atoms_new

def construct_atoms(nums,pos,cell,spinpol,constrts):
    from ase import constraints
    c = None if constrts is None else constraints.FixAtoms(indices=json.loads(constrts))
    a = ase.Atoms(numbers=json.loads(nums),positions=json.loads(pos),cell=json.loads(cell),pbc=[1,1,1]
						,constraint=c)
    if spinpol == 1:
        a.set_initial_magnetic_moments([3 if x in mag_nums else 0 for x in json.loads(nums)])
    return a

"""
