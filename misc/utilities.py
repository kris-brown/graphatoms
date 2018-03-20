# External Modules
from typing import List
import itertools,os,math,subprocess
################################################################################

"""
General functions for manipulating python objects

Cluster-related
    get_cluster
    launch
    safeMkDir

List-related
    flatten

Search-related
    DFS
"""
##################
# Cluster related
#-----------------

def running_jobs_sherlock() -> List[str]:
    """
    Get your current running jobs on the Sherlock cluster
    """
    user = os.environ['USER']

    return subprocess.check_output(['squeue', '-u',user,'-o','%Z']).split()[1:]

def safeMkDir(pth : str ,verbose : bool = True) -> None:
    """
    Make directories even if they already exist
    """
    try:  os.mkdir(pth)
    except OSError:
        if verbose:
            print('directory %s already exists ?!'%pth)

def safeMkDirForce(pth : str) -> None:
    """
    Makes a nested directory, even if intermediate links do not yet exist
    """
    components = pth.split('/')
    curr_dir = [components[0]]
    for c in components[1:]:
        curr_dir.append(c)
        safeMkDir('/'+os.path.join(*curr_dir),verbose=False)


def flatten(lol : List[List]) -> List:
    """
    Flattens a list of Lists to a list
    """
    return [item for sublist in lol for item in sublist]


###############
# Search related
###############
class DFS(object):
    """
    Returns all solutions to a search problem with no cost
    """
    def __init__(self
                ,succ
                ,actions
                ,is_end
                ,result
                )->None:
        #self.init_state = init_state
        self.succ    = succ
        self.actions = actions
        self.is_end  = is_end
        self.result  = result
        self.verbose=False

    def run_dfs(self,s):
        """
        """
        if self.verbose: print('entering run_dfs with s = ',s)
        new_states = [self.succ(s,a) for a in self.actions(s)]
        results = []

        for ns in new_states:
            if self.verbose: print('considering new state = ',ns)
            end = self.is_end(ns)
            if end:
                result = self.result(ns)
                if result is not None:
                    results.append(result)
            else:
                results += self.run_dfs(ns)
        return results
