import sys,os
from subprocess import check_output
#from common     import myexception
#from common_ase import closest_position, get_distance
#from utilities  import get_neighborlist
from ase.io        import read
from ase.visualize import view
import cPickle as pkl
import numpy as np


def get_upper_and_lower_inds(atoms, indices):
    zs = [(index, atoms[index].z) for index in indices]
    zs = sorted(zs, key=lambda x:x[1])
    lowers = zs[:3]
    uppers = zs[-3:]
    lower_inds = [tup[0] for tup in lowers]
    upper_inds = [tup[0] for tup in uppers]
    return lower_inds, upper_inds

def determine_crystal_structure(atoms, ndict=None):
    if ndict == None:
        exclude_list = ['H','N','C','O','P']
        inds = [atom.index for atom in atoms if atom.symbol not in exclude_list]
        ndict = get_neighborlist(atoms, dx=0.6, key_indices=inds, value_indices=inds)
    coords = [len(nlist) for i, nlist in ndict.items()]

    # find most coordinated atom
    # find the mic distance between that atom and all other atoms in "inds" in the unit cell
    # of those atoms, find the closest 14 atoms
    # do a trick like with the deltas to determine bcc or not
    # if it is not bcc, trim down to the closest 12 atoms, and treat that as the nlist going into the  elif maximal_coordination == 12 part

    for i, nlist in ndict.items():
        print i, len(nlist)
    maximal_coordination = max(coords)
    if maximal_coordination == 14:
        return 'bcc'
    elif maximal_coordination == 12:
        # differentiate between hcp and fcc
        max_coord_index = np.argmax(np.array(coords))
        nlist = ndict[max_coord_index]
        lower_inds, upper_inds = get_upper_and_lower_inds(atoms, nlist)


        pos1 = atoms[lower_inds[0]].position
        distances = []
        for upper_atom in upper_inds:
            pos2, _, _ = closest_position(atoms_object=atoms, atom_index=upper_atom, reference_position=pos1)
            pos1_np = np.array(pos1)
            pos2_np = np.array(pos2)
            distances.append(np.linalg.norm(pos1_np - pos2_np))

        distances = sorted(distances)
        deltas = [distances[1] - distances[0], distances[2] - distances[1]]

        if deltas[0] > deltas[1]:
            return 'hcp'
        else:
            return 'fcc'
    else:
        return myexception('expected the most coordinated atom to have 12 or 14-fold coordination. Got ' + str(maximal_coordination))



def main():
    if os.path.exists('../../generated_files/ndicts.pkl') == False:
        generate_ndict = True
    else:
        generate_ndict = False

    # generate_ndict = True
    basepath = '/scratch/users/brohr/vswitch/aayush-slabs'
    crystalstructures = ['fcc','bcc','hcp']

    if generate_ndict:
        print 'generating neighborlists'
        ndicts = {}
        for structure in crystalstructures:
            print structure
            # paths = check_output('find ' + basepath + '/' + structure + ' -name clean -type d', shell=True)
            paths = check_output('find ' + basepath + '/' + structure + ' -name qn.traj', shell=True)
            paths = paths.split('\n')[:-1]
            L = len(paths)
            current_ndict = {}
            for t, path in enumerate(paths):
                print '{}/{}'.format(t, L)
                # atoms = read(path+'/qn.traj')
                atoms = read(path)
                nlist = get_neighborlist(atoms, dx=0.4)
                current_ndict[path] = nlist
            ndicts[structure] = current_ndict

        pkl.dump(ndicts, open('../../generated_files//ndicts.pkl','wb'))
    else:
        ndicts = pkl.load(open('../../generated_files//ndicts.pkl','rb'))
        print 'loaded ndicts.pkl'

    paths = ['/home/brohr/generated_files/qn.traj']

    for path in paths:
        atoms = read(path)
        struct = determine_crystal_structure(atoms)
        print struct
        view(atoms)
