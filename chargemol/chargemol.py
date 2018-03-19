# External
import os,math,time,json,sys,io,shutil,random,ase,glob,string,itertools
import numpy as np
from ase.units import Bohr
from ase.io import write,read
from os.path import exists,join,getctime
# Internal
from graphatoms.misc.utilities import (flatten,get_analysis_folder,get_sherlock_folder
                                   ,safeMkDir,safeMkDirForce,running_jobs_sherlock)
from graphatoms.misc.print_parse  import print_sleep,print_time

"""
FUNCTIONS NEEDED TO GENERATE BONDS.JSON
"""

############
# CONSTANTS
###########
user         = os.environ['USER']
home         = os.environ['HOME']
catalog      = os.environ['CataLogPath']
hostname     = os.environ['HOSTNAME']

if 'sh' in hostname or 'gpu' in hostname:
    sher = os.environ['SHERLOCK']
    if   sher == '1':  host,queue = 'sherlock','iric'
    elif sher == '2':  host,queue = 'sherlock2','suncat'
    else: raise ValueError, 'What cluster is this? '+sher
elif    'su'      in hostname: host = 'suncat'
elif    'nid'     in hostname: host = 'nersc'
else: raise ValueError, 'What cluster is this? '+hostname

conv_dict = {'low':1,       'mid':0.1,     'high': 0.01, 'tom': 0.01}
pw_dict   = {'low':300,     'mid':400,     'high': 500,  'tom': 600}
time_dict = {'low':1,       'mid':2,       'high': 3,    'tom': 4}

logfile   = {'gpaw':'log','qe':'calcdir/log','vasp':'OUTCAR'}
time_dict2= {'gpaw':0.5,    'qe':2,          'vasp':1}

codes     = ['gpaw','vasp']#,'qe']
qualities = ['low']#,'mid','high']
############################################

def test_suite(working_path,trajname):
    """Run a suite of tests on a directory with a traj in it"""

    for code,quality in itertools.product(codes,qualities):
        new_path = join(working_path,code,quality)
        safeMkDirForce(new_path)
        shutil.copyfile(join(working_path,'%s.traj'%trajname)
                        ,join(new_path,'%s.traj'%trajname))

        BondAnalyzer(code,quality).submit(new_path,trajname)

def submit_tom():
    base   = '/scratch/users/ksb/demos/tom_demo/chrgml'

    for root, dirs, files in os.walk(base):
        conditions = ['chargemol_analysis' not in root
                     ,'gpaw' not in root,'vasp' not in root
                    ,'qn.traj' in files]

        if all(conditions):
            vroot,groot = [join(root,x) for x in ['vasp','gpaw']]

            safeMkDir(vroot) ; safeMkDir(groot)
            vbonds,gbonds = [join(x,'chargemol_analysis/qn/bonds.json')
                                                        for x in [vroot,groot]]

            if not os.path.exists(vbonds):
                shutil.copyfile(join(root,'qn.traj'),join(vroot,'qn.traj'))
                BondAnalyzer('vasp','tom').submit(vroot,'qn')

            if not os.path.exists(gbonds):
                shutil.copyfile(join(root,'qn.traj'),join(groot,'qn.traj'))
                BondAnalyzer('gpaw','tom').submit(groot,'qn')

################################################################################
################################################################################
################################################################################

class BondAnalyzer(object):
    """
    dftcode - {'gpaw','qe','vasp'}
    quality - {'low','mid','high'}
    """
    def __init__(self,dftcode='gpaw',quality='low'):
        assert dftcode in ['gpaw','qe','vasp']
        assert quality in ['low','mid','high','tom']

        self.dftcode = dftcode
        self.quality = quality

    # META METHODS
    #--------------------
    def _create_analysis_folder(self,working_path,trajname):
        """
        Creates a directory for job analysis as well as:
        <analysis root>/chargemol_analysis/<trajname>/
        Returns the analysis folder and traj path
        """
        root_folder     = get_analysis_folder(working_path)
        analysis_folder = join(root_folder,'chargemol_analysis',trajname)
        traj_path       = join(working_path,'%s.traj'%trajname)

        safeMkDirForce(analysis_folder)
        shutil.copyfile(traj_path,join(analysis_folder,'%s.traj'%trajname))
        if self.dftcode=='vasp':
            home    = os.environ['HOME']
            vdwpth  = join(home,'../ksb/scripts/vasp/vdw_kernel.bindat')
            shutil.copyfile(vdwpth,join(analysis_folder,'vdw_kernel.bindat'))
            print 'moved from %s to %s'%(vdwpth,analysis_folder)

        return analysis_folder,traj_path

    def _write_metascript(self,analysis_path):
        """
        Write a script which, when sbatch'd (ON SHERLOCK), will perform bond order analysis
        """

        content_dict = {'qe':['python sub_chargemol.py']
                       ,'vasp':['python sub_chargemol.py']
                       ,'gpaw':["NTASKS=`echo $SLURM_TASKS_PER_NODE|tr '(' ' '|awk '{print $1}'`"
                               ,"NNODES=`scontrol show hostnames $SLURM_JOB_NODELIST|wc -l`"
                               ,'NCPU=`echo " $NTASKS * $NNODES " | bc`'
                               ,'source /scratch/users/ksb/gpaw/paths.bash'
                               ,'mpirun -n $NCPU gpaw-python sub_chargemol.py']}

        hours  = time_dict[self.quality] * time_dict2[self.dftcode]

        script = '\n'.join(['#!/bin/bash'
                        ,'#SBATCH -p %s,owners'%queue
                        ,'#SBATCH --time=%s:00'%(print_time(hours))
                        ,'#SBATCH --mem-per-cpu=4000'
                        ,'#SBATCH --error=err.log'
                        ,'#SBATCH --output=opt.log'
                        ,'#SBATCH --nodes=1'
                        ,'#SBATCH --ntasks-per-node=16']
                        + content_dict[self.dftcode])

        path = join(analysis_path,'sub_chargemol.sh')
        with open(path,'w') as f: f.write(script)
        os.system('chmod 755 ' + path)

    def _write_script(self,analysis_path,working_path,trajname):
        script = ('from CataLog.chargemol.chargemol import BondAnalyzer\n'
                +"BondAnalyzer('%s','%s')"%(self.dftcode,self.quality)
                +".analyze('%s','%s')"%(working_path,trajname))

        with open(join(analysis_path,'sub_chargemol.py'),'w') as f:
            f.write(script)

    # MAIN PIPELINE METHODS
    #----------------------

    def _generate_charge_density(self,analysis_path,atoms_path):
        """
        """
        # Initialize
        #-----------
        os.chdir(analysis_path)
        atoms = read(atoms_path)
        atoms.set_pbc([1,1,1])

        # Run DFT
        #--------
        calc_dict = {'qe':self._mk_qe,'vasp':self._mk_vasp,'gpaw':self._mk_gpaw}
        calc      = calc_dict[self.dftcode](atoms_path)
        atoms.set_calculator(calc)
        atoms.get_potential_energy()

        # Write charge density files if necessary
        #-----------------------------------------
        if self.dftcode == 'gpaw':
            density = calc.get_all_electron_density() * Bohr**3
            write(join(analysis_path,'total_density.cube')
                 ,atoms,data=density)

        elif self.dftcode == 'qe': # REQUIRES HACK OF ESPRESSO
            calc.cube_charge_density('valence_density.cube')
            #calc.extract_charge_density('valence_density.cube')
            #extra_charge = get_extra_charge(analysis_path)
            #print 'normalization factor = ',extra_charge
            #normalize_cube('valence_density.cube',extra_charge)

    def _write_chargemol_input(self,analysis_path, _ ):
        """Write job_control.txt to an analysis folder"""
        if self.dftcode=='qe':
            core_dict = self._core_dict(analysis_path)
            core = '\n'.join(['<number of core electrons>'
                             ,'\n'.join(['%d %d'%(k,v) for k,v in core_dict.items()])
                             ,'</number of core electrons>'])
        else: core = ''

        job_control = '\n'.join(['<net charge>',"0.0","</net charge>"
                                ,"<periodicity along A, B, and C vectors>"
                                ,".true.",".true.",".true."
                                ,"</periodicity along A, B, and C vectors>"
                                ,'<compute BOs>','.true.','</compute BOs>'
                                ,'<atomic densities directory complete path>'
                                ,'{0}/chargemol/atomic_densities/'.format(catalog)
                                ,'</atomic densities directory complete path>'
                                ,'<charge type>','DDEC6','</charge type>'
                                ,core])

        with open(join(analysis_path,'job_control.txt'),'w') as f:
            f.write(job_control)

    def _call_chargemol(self,analysis_path, _ ):
        """Call chargemol binary from a specified folder"""
        os.chdir(analysis_path)
        path_to_chargemol = join(home,'CataLog','chargemol','chargemol_binary') # need to compile parallel with relaxed tolerance but getting error
        os.system(path_to_chargemol)
        print 'executing: ',path_to_chargemol
        check = join(analysis_path,'DDEC6_even_tempered_bond_orders.xyz')
        if not exists(check):
            pass
            #c = get_extra_charge(analysis_path)
            #print 'normalizing by factor of ',c
            #normalize_cube('valence_density.cube',c)
            #os.system(path_to_chargemol)

    def _postprocess(self,analysis_path,atoms_path):
        if self.dftcode=='vasp': sortdict = self._sort_dict(analysis_path)
        else:                    sortdict = {}

        potential_edges = parse_chargemol(analysis_path,atoms_path,sortdict)  # parse output

        data = [e.__dict__ for e in potential_edges]
        with open(join(analysis_path,'bonds.json'),'w') as f: json.dump(data,f)

    def _write_metadata(self,analysis_path, _ ):
        """Write data about how bonds.json was calculated to file"""
        pth      = join(analysis_path,'metadata.json')
        ctime    = getctime(join(analysis_path,logfile[self.dftcode]))
        metadata = {'timestamp':time.time(),'user':user
                    ,'dftcode':self.dftcode,'quality':self.quality
                    ,'time':time.time() - ctime}
        with open(pth,'w') as f: json.dump(metadata,f)

    # DFTCODE-SPECIFIC METHODS
    #-------------------------
    def _mk_kpts(self,atoms_path):
        """Chooses appropriate kpt spacing"""
        if self.quality == 'tom':
            print 'using 441 kpts because TOM'
            return [4,4,1]

        def get_kpt(x): return int(math.ceil(15/np.linalg.norm(x)))
        cell = read(atoms_path).get_cell()
        return  map(get_kpt,cell)

    def _mk_gpaw(self,atoms_path):
        """Make a GPAW calculator"""
        import gpaw,gpaw.poisson
        from gpaw.utilities import h2gpts
        cell  = read(atoms_path).get_cell()
        scale = conv_dict[self.quality]
        return gpaw.GPAW(mode='lcao',basis = 'dzp',mixer=gpaw.Mixer(0.1, 5, 100)
                        ,gpts          = h2gpts(0.15, cell, idiv=8)
                        ,poissonsolver = gpaw.poisson.PoissonSolver(relax='GS',eps=1e-10)
                        ,symmetry      = {'do_not_symmetrize_the_density': True}
                        ,txt           = 'log'
                        ,convergence   = {'energy':0.1 * scale
                                         ,'density':0.01 * scale
                                         ,'bands':-10})
    def _mk_vasp(self,atoms_path):
        """Make a VASP calculator"""
        import ase.calculators.vasp as vasp_calculator

        if self.quality == 'tom': xc = 'beef-vdw'
        else:                     xc = 'PBE'

        return vasp_calculator.Vasp(encut = pw_dict[self.quality] ,xc = xc
                                    ,kpts = self._mk_kpts(atoms_path),npar = 1
                                    ,algo= 'fast',prec = 'accurate',nsw =  0
                                    ,nelmdl=3 ,isym= 0 ,lcharg = True,laechg = True)

    def _mk_qe(self,atoms_path):
        """Make a QE calculator"""
        from espresso import espresso
        return espresso( pw      = pw_dict[self.quality]
                        ,dw      = pw_dict[self.quality] * 10
                        ,xc      = 'BEEF'
                        ,kpts    = self._mk_kpts(atoms_path)
                        ,nbands  = -20
                        ,sigma   = 0.1
                        ,dipole  = {'status':True} # CHANGE THIS BACK
                        ,psppath = '/home/vossj/suncat/psp/gbrv1.5pbe'
                        ,spinpol = False
                        ,convergence = {'energy':      1e-3 * conv_dict[self.quality]
                                       ,'mixing':      0.1
                                       ,'mixing_mode': 'plain'
                                       ,'maxsteps':    200}
                        ,outdir='calcdir')

    def _sort_dict(self,analysis_path):
        """
        Read ase-sort.dat. Only should be called if VASP
        """
        with open(join(analysis_path,'ase-sort.dat'),'r') as f:
            loglines = map(lambda x: map(int,x.split()),f.readlines())
            return dict(map(tuple,loglines))

    def _core_dict(self,analysis_path):
        """
        For QE calculations, get {atomic number : # core electrons}
        """
        cordict = {}
        with open(join(analysis_path,'calcdir','log'),'r') as f:
            loglines = filter(None,map(string.split,f.readlines()))
            for i,line in enumerate(loglines):
                if line[0] == 'PseudoPot.':                      # for each PSP:
                    elem = line[4]                               # symbol
                    num  = ase.data.chemical_symbols.index(elem) # atomic number
                    val  = int(float((loglines[i+3][-1])))       # valence electrons
                    cordict[num] = num - val                     # core electrons
        return cordict

    # EXPOSED METHODS
    #-----------------
    def submit(self,working_path,trajname):
        analysis_path,trajpth= self._create_analysis_folder(working_path,trajname)
        checks = [analysis_path in running_jobs_sherlock()
                 ,exists(join(analysis_path,'bonds.json'))]
        if not any(checks):
            self._write_metascript(analysis_path)
            self._write_script(analysis_path,working_path,trajname)
            os.chdir(analysis_path)
            os.system('sbatch %s'%join(analysis_path,'sub_chargemol.sh'))

    def analyze(self,working_path,trajname,start_ind=0):
        """Start the process at any point"""

        analysis_path,trajpth= self._create_analysis_folder(working_path,trajname)

        pipeline = [self._generate_charge_density # 0
                   ,self._write_chargemol_input   # 1
                   ,self._call_chargemol          # 2
                   ,self._postprocess             # 3
                   ,self._write_metadata]         # 4

        for process in pipeline[start_ind:]:
            process(analysis_path,trajpth)

    def launch_neb(self,neb_path):
        """Take a neb folder to generate chargemol jobs for all images"""
        os.chdir(neb_path)
        for neb in glob.glob('neb?.traj'): self.submit(neb_path,neb)

    def launch_catalog(self,constraints=[],limit=1, array=False):
        """
        Randomly attempt bond order analysis on jobs in a user's directory if
        it hasn't already been completed
        """
        from CataLog.datalog.db_utils  import Query
        from CataLog.datalog.manageDB  import update_db
        counter = limit # we don't want to limit the Query because all threads would have the same job
        if array:
            print_sleep(random.randint(0,60)) # randomly wait so that all array jobs don't query at same time
        update_db('chargemol',load=False,retry=True,verbose=False)
        folders = Query(constraints=constraints+[RELAXORLAT_,Not(CHARGEMOL)]
                            ,table=job,order=Random(),limit=limit).query_col(STORDIR)


        assert len(folders)==limit, '%d != limit %d '%(len(folders),limit)
        for folder in folders:
            for t in ['init','final']:
                self.submit(get_sherlock_folder(folder),t)

################################################################################
################################################################################
################################################################################

####################
# Parsing Classes
#--------------------
class BondOrderSection(object):
    """Process one section of the Bond Order output of Chargemol"""
    def __init__(self,atoms,ind,sumBO,raw_lines,sortdict,pbcdict):
        self.atoms = atoms                     # ase.Atoms
        self.ind   = ind                       # Int
        self.sumBO = sumBO                     # Float
        self.bonds = map(parse_line,raw_lines) # [(index,bo,offset)]
        self.sortdict = sortdict
        self.pbcdict  = pbcdict

    def _relative_shift(self,i,j):
        """
        Given a pbc_dict and two indices, return the original pbc shift
        for a bond from i to j
        """
        pi,pj = [np.array(self.pbcdict[x]) for x in [i,j]]
        return pj - pi

    def makeEdge(self,(toInd,bo,offset)):
        """
        Creates an Edge instance from the result of a parsed Bond Order log line
        """
        fromInd = self.ind

        if self.sortdict: # undo VASP reordering if necessary
            fromInd,toInd = map(self.sortdict.get,[fromInd,toInd])

        # correct for WRAPPING atoms
        offset = (np.array(offset) +  self._relative_shift(fromInd,toInd)).tolist()

        shift = np.dot(offset, self.atoms.get_cell()) # PBC shift
        p1    = self.atoms[fromInd].position
        p2    = self.atoms[toInd].position + shift
        d     = np.linalg.norm(p2-p1)

        return PotentialEdge(fromInd,toInd,d,offset,bo,self.sumBO)

    def make_edges(self):
        """Apply edgemaker to result of parsing logfile lines"""
        return  map(self.makeEdge,self.bonds)

class PotentialEdge(object):
    """
    Container for information we need to decide later if it's a
    graph-worthy bond. This class is nothing more than a dictionary.
    """
    def __init__(self,fromNode,toNode,distance,offset,bond_order,tot):
        self.fromNode = fromNode            # Int
        self.toNode   = toNode              # Int
        self.distance = round(distance,2)   # Float
        self.offset   = offset              # (Int,Int,Int)
        self.bondorder= round(bond_order,3) # Float
        self.total_bo = round(tot,2)        # Float

###################
# Parsing Functions
#-------------------

def parse_chargemol(analysis_path,atoms_path,sortdict):
    """
    Read file DDEC6_even_tempered_bond_orders.xyz
    """
    header,content,sections,head_flag,counter, = [],[],[],True,-1       # Initialize
    filepath = join(analysis_path,'DDEC6_even_tempered_bond_orders.xyz')# File to parse

    atoms = read(atoms_path)
    pbcdict = mk_pbc_dict(atoms) # the 'true' PBC coordinate of each atom

    with open(filepath,'r') as f: lines = f.readlines()
    for line in lines:
        if not (line.isspace() or line[:4] == ' 201'): # remove blank lines and calendar date
            if line[1]=='=':
                head_flag = False                           # no longer in header  section
                chargemol_pbc = parse_chargemol_pbc(header,atoms.get_cell()) # PBC of chargemol atoms

            elif head_flag: header.append(line)     # we're still in the header section
            elif 'Printing' in line:                # we're describing a new atom
                content = []                        # reset buffer content
                counter += 1                        # update index of our from_index atom
            elif 'sum' in line:                     # summary line at end of a section
                sumBO = float(line.split()[-1])     # grab sum of bond orders
                sections.append(BondOrderSection(atoms,counter,sumBO,content
                                                ,sortdict,dict_diff(pbcdict,chargemol_pbc)))
            else: content.append(line)              # business-as-usual, add to buffer

    return  flatten([x.make_edges() for x in sections]) # single list of edges

def parse_line(line):
    """
    Get bonded atom, bond order, and offset
    """
    assert line[:16]==' Bonded to the (', "No parse line ->"+line+'<- %d'%len(line)
    offStr,indStr,boStr = line[16:].split(')')
    offset = map(int,offStr.split(',')) # Chunk containing offset info
    ind    = int(indStr.split()[-3]) - 1        # Chunk containing index info (chargemol starts at 1, not 0)
    bo     = float(boStr.split()[4])            # Chunk containing B.O. info
    return (ind,bo,offset)

####
# Misc
#-----

def parse_chargemol_pbc(header_lines,cell):
    atoms = ase.Atoms(cell=cell)
    for i,l in enumerate(header_lines[2:]):
        try:
            s,x,y,z,_ = l.split()
            p = map(float,[x,y,z])
            atoms.append(ase.Atom(s,position=p))
        except Exception as e: pass
    return mk_pbc_dict(atoms)


def mk_pbc_dict(atoms):
    def f(x):
        if   x < 0: return -1
        elif x < 1: return 0
        else:       return 1

    scaled_pos = zip(range(len(atoms)),atoms.get_scaled_positions(wrap=False).tolist())
    pbc_dict   = {i : map(f,p) for i,p in scaled_pos}
    return pbc_dict

def dict_diff(d1,d2):
    return {i: np.array(d2[i]) - np.array(d1[i]) for i in d1.keys()}


"""
Invoke these if we get quantum espresso errors that exceed the new 0.2 tolerance
def get_extra_charge2(analysis_path):
    #gets extra charge from qe log file
    with open(join(analysis_path,'calcdir/log'),'r') as f: lines = f.readlines()
    for line in lines:
        if 'renormalised' in line:
            _,_,n,_,_,d = line.split()
            return float(n[:-1]) / float(d)
def get_extra_charge(analysis_path):
    #gets extra charge from failed chargemol analysis file
    with open(join(analysis_path,'valence_cube_DDEC_analysis.output'),'r') as f:
        lines = f.readlines()
        n,d = 0,0
        for line in lines:
            if 'nvalence' in line:                 n = float(line.split()[-1])
            elif 'numerically integrated' in line: d = float(line.split()[-1])

        if n==0 or d ==0: print 'weird Chargemol error!'; sys.exit()
        else: return n / d
def normalize_cube(cube_path,constant):
    from ase.io.cube import read_cube_data
    data, atoms = read_cube_data(cube_path)
    write(cube_path,atoms,data=data*constant)
"""
