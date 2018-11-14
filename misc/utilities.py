# External Modules
import itertools,os,subprocess
################################################################################

"""
General functions for manipulating python objects
------------------------------------------------
General
    true
    negate
Cluster-related
    get_cluster
    launch
    safeMkDir

List-related
    flatten

Dict-related
    merge_dicts


Search-related
    DFS
"""

def true(_):   return True
def negate(x): return -x
##################
# Cluster related
#-----------------

def running_jobs_sherlock():
    """
    Get your current running jobs on the Sherlock cluster
    """
    user = os.environ['USER']

    return subprocess.check_output(['squeue', '-u',user,'-o','%Z']).split()[1:]

def safeMkDir(pth ,verbose ) :
    """
    Make directories even if they already exist
    """
    try:  os.mkdir(pth)
    except OSError:
        if verbose:
            print('directory %s already exists ?!'%pth)

def safeMkDirForce(pth) :
    """
    Makes a nested directory, even if intermediate links do not yet exist
    """
    components = pth.split('/')
    curr_dir = [components[0]]
    for c in components[1:]:
        curr_dir.append(c)
        safeMkDir('/'+os.path.join(*curr_dir),verbose=False)


def flatten(lol ):
    """
    Flattens a list of Lists to a list
    """
    return [item for sublist in lol for item in sublist]

################
# Search related
################

def merge_dicts(listDicts) :
    """
    Merge dictionaries, presumes no overlap in keys
    """
    return dict(itertools.chain.from_iterable([x.items() for x in listDicts]))

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
                ) :
        self.succ    = succ
        self.actions = actions
        self.is_end  = is_end
        self.result  = result
        self.verbose = False

    def run_dfs(self,s):
        """
        Run DFS from some (starting or intermediate) State until termination
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
