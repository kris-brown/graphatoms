#External Modules
from typing import Dict
import json,math
import ase          # type: ignore
import numpy as np  # type: ignore
################################################################################
"""
Functions for interacting with ASE

Element Lists
    nonmetal_symbs
    nonmetals

Geometry
    angle
    dihedral

Misc
    match_indices

"""

# Keld Data Related
#-----------------


# Geometry

def angle(v1 : np.array,v2 : np.array) -> float:
    return np.arccos(np.dot(v1,np.transpose(v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def dihedral(v1: np.array,v2: np.array,v3: np.array)->float:
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
    theta     = -math.atan2(sin_theta,cos_theta)
    return theta


##############
# Element lists
###############
nonmetal_symbs   = ['H','C','N','P','O','S','Se','F','Cl','Br'
                   ,'I','At','He','Ne','Ar','Kr','Xe','Rn']

nonmetals        = [ase.data.chemical_symbols.index(x) for x in nonmetal_symbs]

##############
# Misc
###############

def match_indices(atoms1 : ase.Atoms,atoms2: ase.Atoms) -> Dict[int,int]:
    """
    Returns mapping dictionary that maps indices in atoms2 to be as 'close'
    to atoms1 as possible
    """
    from scipy.spatial.distance import cdist  # type: ignore

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

def test_match() -> None:
    """
    Demonstrates the match_indices function on a 1x1x6 slab with its bottom atom
    moved to the end of the index list
    """
    from ase.io import read # type: ignore
    a1 = read('/scratch/users/ksb/fireworks/jobs/relax_2017_11_04_11_05_10/final.traj')
    a2 = read('/scratch/users/ksb/fireworks/jobs/relax_2017_11_04_11_05_10/init.traj')
    at1 = a2.pop(0) # make sure switch respects constraints!
    a2.append(at1)  # elements of a2 are now reordered
    print('a1 (constrained = %s)'%(a1.constraints[0]))
    for a in a1: print(a)
    print('\na2 (constrained = %s)'%(a2.constraints[0]))
    for a in a2: print(a)
    print('result: ',match_indices(a1,a2))
    # should be {0 -> 5, 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4}
