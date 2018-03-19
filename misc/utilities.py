# External Modules
import itertools,os,math,subprocess
################################################################################

"""
General functions for manipulating python objects

General
    identity

Cluster-related
    get_cluster
    get_analysis_folder
    get_sherlock_folder
    launch
    safeMkDir

String Related
    modify_time
    sub_binds
    replacer

List-related
    partition
    chunks
    lol
    flatten
    gcd
    normalize_list

Antiquated
    change_adsorbatedict_to_indice_list
"""
##########
# General
#---------
identity = lambda x : x
negate   = lambda x: -x
true     = lambda x : True
user     = os.environ['USER']
##################
# Cluster related
#-----------------

def running_jobs_sherlock():
    return subprocess.check_output(['squeue', '-u',user,'-o','%Z']).split()[1:]


def get_cluster():
    """Get the name of the cluster this command is being executed on"""
    hostname = os.environ['HOSTNAME'].lower()
    if      'sh'    in hostname: return 'sherlock'
    elif   'gpu-15' in hostname: return 'sherlock'
    elif    'su'    in hostname: return 'suncat' #important to distinguish suncat2 and 3?
    elif    'kris'  in hostname: return 'kris'
    else: raise ValueError, "getCluster did not detect SH or SU in %s"%hostname

def get_analysis_folder(sd):
    """Takes either the stordir or the sherlock folder and gives analysis folder"""
    sd2 = get_sherlock_folder(sd)
    if  'suncat' in sd2: sd2=sd2.replace('suncat_jobs_copy/jobs/','analysis_suncat/')
    elif 'nersc' in sd2: sd2=sd2.replace('nersc_jobs_copy/jobs/','analysis_nersc/')
    else:              sd2=sd2.replace('jobs/','analysis_sherlock/')
    return sd2

def get_sherlock_folder(sd):
    """Takes a storage_directory and gives the path to its copy on sherlock"""
    replace_dict = {'/nfs/slac/g/suncatfs/ksb/share/jobs/' : '/scratch/users/ksb/share/suncat_jobs_copy/jobs/'
                   ,'/global/cscratch1/sd/krisb/share/jobs/':'/scratch/users/ksb/share/nersc_jobs_copy/jobs/'}
    return replacer(str(sd),replace_dict)


def launch():
    """Tell FireWorks to submit READY jobs to their respective queues"""
    os.system('$HOME/CataLog/fw/launcher.sh')

def safeMkDir(pth,verbose=True):
    """Make directories even if they already exist"""
    try:            os.mkdir(pth)
    except OSError:
        if verbose: print 'directory %s already exists ?!'%pth

def safeMkDirForce(pth):
    """Makes a nested directory, even if intermediate links do not yet exist"""
    components = pth.split('/')
    curr_dir = [components[0]]
    for c in components[1:]:
        curr_dir.append(c)
        safeMkDir('/'+os.path.join(*curr_dir),verbose=False)

#############
# SQL Related
#-----------

################
# String related
#--------------
def modify_time(t,multiplier = 2):
    def printTime(floatHours):
    	intHours = int(floatHours)
    	return "%02d:%02d" % (intHours,(floatHours-intHours)*60)

    """Modifies time in either HH:MM::SS or HH:MM format. Min time = 1 hr, max time = 40 hr"""
    times = [int(x) for x in t.split(':')]
    HHMMSS = len(times) == 3
    tot = times[0]+times[1]/60.0 + (times[2]/3600.0 if HHMMSS else 0)
    return printTime(min(40,math.ceil(multiplier*tot)))+(':00' if HHMMSS else '')

def sub_binds(sql_select):
    """Prints a sql command in a human-readable way that can be copypasted
    into DB Browswer for SQLite."""

    keywords = ['INNER','FROM','HAVING','WHERE',"GROUP BY",", "]

    (sql_command,binds) = tuple(sql_select)

    for b in binds: sql_command=sql_command.replace('?',repr(b),1)

    replace_dict = {x:('\n\t'+x) for x in keywords}

    print '\n\t'+replacer(sql_command,replace_dict)+'\n'

def replacer(s,replace_dict):
    """Executes a series of string replacement operations, specified by a
    dictionary"""
    for k,v in replace_dict.items(): s = s.replace(k,v)
    return s

#############
# Dict related
#---------------
def dict_diff(first, second):
    """ Return a dict of keys that differ with another config object.  If a value is
        not found in one fo the configs, it will be represented by KEYNOTFOUND.
        @param first:   Fist dictionary to diff.
        @param second:  Second dicationary to diff.
        @return diff:   Dict of Key => (first.val, second.val)
    """
    KEYNOTFOUND = '<KEYNOTFOUND>'       # KeyNotFound for dictDiff
    diff = {}
    # Check all keys in first dict
    for key in first.keys():
        if (not second.has_key(key)):
            diff[key] = (first[key], KEYNOTFOUND)
        elif (first[key] != second[key]):
            diff[key] = (first[key], second[key])
    # Check all keys in second dict to find missing
    for key in second.keys():
        if (not first.has_key(key)):
            diff[key] = (KEYNOTFOUND, second[key])
    return diff


def merge_dicts(listDicts): return dict(itertools.chain.from_iterable([x.items() for x in listDicts])) #presumes no overlap in keys

##############
# List related
#-------------
def chunks(l, n): #Yield successive n-sized chunks from l.
    for i in range(0, len(l), n): yield l[i:i + n]

def flatten(lol): return [item for sublist in lol for item in sublist] #flattens a List Of Lists to a list

def gcd(args):
    """Greatest common denominator of a list"""
    if len(args) == 1:  return args[0]
    L = list(args)
    while len(L) > 1:
        a,b = L[len(L) - 2] , L[len(L) - 1]
        L = L[:len(L) - 2]
        while a:   a, b = b%a, a
        L.append(b)
    return abs(b)

def normalize_list(l):
    """ [a,a,a,a,b,b,c,c] => [a,a,b,c] """
    if len(l)==0: return l
    d   = {x:l.count(x) for x in l}
    div = gcd(d.values())
    norm= [[k]*(v/div) for k,v in d.items()]
    return [item for sublist in norm for item in sublist]


############
# Antiquated
#-----------

def change_adsorbatedict_to_indice_list(usr):
    import CataLog.datalog.databaseFuncs as db
    import json
    from CataLog.misc.sql_utils import AND
    import CataLog.datalog.databaseFuncs as db
    import numpy as np
    from manageSharedDatabase import sync_suncat

    ads_num_atoms_dict = {'NN':2,'NNH':3,'NNH2':4,'NNH3':5,'N':1,'NH':2,'NH2':3,'NH3':4,'H':1,'OH':2,'O':1}
    #Sync the database so that slac and sherlock share are up to date
    sync_suncat()

    # Replace {'NH':[pos]} -> [[39, 40, 41]] for adsorbed surfaces
    #Make query by grabbing jobs with non-empty adsorbate dictionaries
    query = db.Query(job.id, job.adsorbates, atoms.numbers['job_id','adsorbates_ksb','final.numbers'],AND('adsorbates_ksb != \'{}\'','adsorbates_ksb is not null','user = \''+usr+'\''))
    #Filter out non dictionary adsorbate values
    output  = filter(lambda row: isinstance(json.loads(row[2]),dict),query.query())
    output = zip(*output)

    #If jobs meet these criteria perform the transformation to the indice list
    if len(output)>0:
        job_ids,ads_dicts,numbers = output        #extract the relevant data from the query
        #Turn the adsorbate key into a number of adsorbate atoms
        ads_dicts = map(lambda dic: ads_num_atoms_dict.get(dic.keys()[0]), map(json.loads, ads_dicts))
        num_atoms = map(len,map(json.loads, numbers))        #Find the number of atoms in the final atoms object
        #Combine the number of adsorbate atoms and the number of total atoms into an adsorbate indice list
        #THIS ASSUMES ADSORBATE WAS LAST THING ADDED TO TRAJ
        indices_list = map(lambda ads_count, atom_count: str([list(np.arange(atom_count-ads_count,atom_count))]),ads_dicts,num_atoms)
        #Modify storage directories
        for i, new_ads_list in zip(job_ids,indices_list):
            db.modify_storage_directory('job_id =%d'%i,'result.json','adsorbates',new_ads_list,dict_loc = 'params')
            db.modify_storage_directory('job_id =%d'%i,'params.json','adsorbates',new_ads_list,dict_loc = 'params')

    # Replace {} with [] for bare surfaces
    #Make query of bare surfaces with adsorbate dictionaries {}
    query = db.Query(['job_id','storage_directory','adsorbates_ksb','final.numbers'],AND('adsorbates_ksb = \'{}\'','adsorbates_ksb is not null','user = \''+usr+'\''))
    output  = filter(lambda row: isinstance(json.loads(row[2]),dict),query.query())
    output = zip(*output)
    if len(output)>0:    #If jobs match this criteria modify the storage directories
        job_ids = output[0]
        for i in zip(job_ids):
            db.modify_storage_directory('job_id =%d'%i,'result.json','adsorbates','[]',dict_loc = 'params')
            db.modify_storage_directory('job_id =%d'%i,'params.json','adsorbates','[]',dict_loc = 'params')

    #Sync sherlock and suncat
    syncSuncat()
