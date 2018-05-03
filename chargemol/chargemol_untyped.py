# External
import os,math,time,json,sys,io,shutil,random,glob,string,itertools,subprocess
from os.path    import exists,join,getctime

import ase                          #type: ignore
import numpy as np                  #type: ignore
from ase.units  import Bohr         #type: ignore
from ase.io     import write,read   #type: ignore


def print_time(floatHours ):
	intHours = int(floatHours)
	return "%02d:%02d" % (intHours,(floatHours-intHours)*60)

def running_jobs_sherlock() :
    """
    Get your current running jobs on the Sherlock cluster
    """
    user = os.environ['USER']

    return subprocess.check_output(['squeue', '-u',user,'-o','%Z']).split()[1:]

def safeMkDir(pth  ,verbose =True):
    """
    Make directories even if they already exist
    """
    try:  os.mkdir(pth)
    except OSError:
        if verbose:
            print('directory %s already exists ?!'%pth)

def safeMkDirForce(pth ) :
    """
    Makes a nested directory, even if intermediate links do not yet exist
    """
    components = pth.split('/')
    curr_dir = [components[0]]
    for c in components[1:]:
        curr_dir.append(c)
        safeMkDir('/'+os.path.join(*curr_dir),verbose=False)

def flatten(lol) :
    """
    Flattens a list of Lists to a list
    """
    return [item for sublist in lol for item in sublist]

"""
FUNCTIONS NEEDED TO GENERATE BONDS.JSON
"""

############
# CONSTANTS
###########
user         = os.environ['USER']
home         = os.environ['HOME']
graphatoms   = os.environ['GRAPHATOMS_PATH']
queue        = 'iric' 


conv_dict = {'low':1,       'mid':0.1,     'high': 0.01, 'tom': 0.01}
pw_dict   = {'low':300,     'mid':400,     'high': 500,  'tom': 600}
time_dict = {'low':1,       'mid':2,       'high': 3,    'tom': 4}

logfile   = {'gpaw':'log','qe':'calcdir/log','vasp':'OUTCAR'}
time_dict2= {'gpaw':0.5,    'qe':2,          'vasp':0.2}

codes     = ['gpaw','vasp']#,'qe']
qualities = ['low']#],'mid','high']
############################################

def test_suite(working_path
              ,trajname
              ) :
    """
    Run a suite of tests on a directory with a traj in it
    """

    for code,quality in itertools.product(codes,qualities):
        new_path = join(working_path,code,quality)
        safeMkDirForce(new_path)
        shutil.copyfile(join(working_path,'%s.traj'%trajname)
                        ,join(new_path,'%s.traj'%trajname))

        BondAnalyzer(code,quality).submit(new_path,trajname)


################################################################################
################################################################################

class BondAnalyzer(object):
    """
    Parameterizes a set of methods that manipulate Atoms objects and produce
    a file (bond.json) that contains information needed to construct a graph

    Parameters:
    dftcode - {'gpaw','qe','vasp'}
    quality - {'low','mid','high','tom'}
    """

    def __init__(self
                ,dftcode  ='gpaw'
                ,quality  = 'low'
                ) :

        assert dftcode in ['gpaw','qe','vasp']
        assert quality in ['low','mid','high','tom']
        if dftcode == 'vasp' and quality == 'tom':  # we are using BEEF functional
            assert 'VASP_VDW_KERNEL' in os.environ

        self.dftcode = dftcode
        self.quality = quality

    # META METHODS
    #--------------------
    def _create_analysis_folder(self
                               ,root_folder
                               ,trajname
                               ) :
        """
        Creates a directory for job analysis as well as:
        <root_folder>/chargemol_analysis/<trajname>/
        Returns the analysis folder and traj path
        """
        analysis_folder = join(root_folder,'chargemol_analysis',trajname)
        traj_path       = join(root_folder,'%s.traj'%trajname)

        safeMkDirForce(analysis_folder)
        shutil.copyfile(traj_path,join(analysis_folder,'%s.traj'%trajname))

        if self.dftcode=='vasp' and self.quality=='tom':
            home    = os.environ['HOME']
            vdwpth  = join(home,'../ksb/scripts/vasp/vdw_kernel.bindat')
            shutil.copyfile(vdwpth,join(analysis_folder,'vdw_kernel.bindat'))

        return analysis_folder,traj_path

    def _write_metascript(self
                         ,analysis_path
                         ):
        """
        Write a script which, when sbatch'd (ON SHERLOCK), will perform bond order analysis
        """

        content_dict = {'qe':['source %s/.env/bin/activate'%graphatoms
                             ,'python3 sub_chargemol.py']
                       ,'vasp':['source %s/.env/bin/activate'%graphatoms
                               ,'export PYTHONPATH=/scratch/users/ksb'
                               ,'python3 sub_chargemol.py']
                       ,'gpaw':["NTASKS=`echo $SLURM_TASKS_PER_NODE|tr '(' ' '|awk '{print $1}'`"
                               ,"NNODES=`scontrol show hostnames $SLURM_JOB_NODELIST|wc -l`"
                               ,'NCPU=`echo " $NTASKS * $NNODES " | bc`'
                               ,'source /scratch/users/ksb/gpaw/paths.bash'
                               ,'source %s/.env/bin/activate'%graphatoms
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

    def _write_script(self
                     ,analysis_path
                     ,working_path
                     ,trajname
                     ):
        """
        Helpful docstring
        """
        script = ('from graphatoms.chargemol.chargemol import BondAnalyzer\n'
                +"BondAnalyzer('%s','%s')"%(self.dftcode,self.quality)
                +".analyze('%s','%s')"%(working_path,trajname))

        with open(join(analysis_path,'sub_chargemol.py'),'w') as f:
            f.write(script)

    # MAIN PIPELINE METHODS
    #----------------------

    def _generate_charge_density(self
                                ,analysis_path
                                ,atoms_path
                                ) :
        """
        Helpful docstring
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

    def _write_chargemol_input(self
                              ,analysis_path
                              ,atoms_path
                              ) :
        """
        Write job_control.txt to an analysis folder
        """

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
                                ,'{0}/chargemol/atomic_densities/'.format(graphatoms)
                                ,'</atomic densities directory complete path>'
                                ,'<charge type>','DDEC6','</charge type>'
                                ,core])

        with open(join(analysis_path,'job_control.txt'),'w') as f:
            f.write(job_control)

    def _call_chargemol(self
                       ,analysis_path
                       ,atoms_path
                       ) :
        """
        Call chargemol binary from a specified folder
        """

        os.chdir(analysis_path)
        path_to_chargemol = join(home,'graphatoms','chargemol','chargemol_binary') # need to compile parallel with relaxed tolerance but getting error
        os.system(path_to_chargemol)
        print('executing: ',path_to_chargemol)
        check = join(analysis_path,'DDEC6_even_tempered_bond_orders.xyz')
        if not exists(check):
            pass
            #c = get_extra_charge(analysis_path)
            #print 'normalizing by factor of ',c
            #normalize_cube('valence_density.cube',c)
            #os.system(path_to_chargemol)

    def _postprocess(self
                    ,analysis_path
                    ,atoms_path
                    ) :
        """
        Helpful docstring
        """

        if self.dftcode=='vasp': sortdict = self._sort_dict(analysis_path)
        else:                    sortdict = {}

        potential_edges = parse_chargemol(analysis_path,atoms_path,sortdict)  # parse output

        data = [e.__dict__ for e in potential_edges]
        with open(join(analysis_path,'bonds.json'),'w') as f: json.dump(data,f)

    def _write_metadata(self
                       ,analysis_path
                       ,atoms_path
                       ) :
        """
        Write data about how bonds.json was calculated to file
        """

        pth      = join(analysis_path,'metadata.json')
        ctime    = getctime(join(analysis_path,logfile[self.dftcode]))
        metadata = {'timestamp':time.time(),'user':user
                    ,'dftcode':self.dftcode,'quality':self.quality
                    ,'time':time.time() - ctime}

        with open(pth,'w') as f:
            json.dump(metadata,f)

    # DFTCODE-SPECIFIC METHODS
    #-------------------------
    def _mk_kpts(self
                ,atoms_path
                ):
        """
        Chooses appropriate kpt spacing
        """
        # handle special case
        if self.quality == 'tom':
            print('using 441 kpts because TOM')
            return [4,4,1]

        def get_kpt(x ) :
            return int(math.ceil(15/np.linalg.norm(x)))

        cell = read(atoms_path).get_cell() # 3 x 3 array

        return  [get_kpt(v) for v in cell]

    def _mk_gpaw(self
                ,atoms_path
                ) : # ignore return type ... not everyone can import GPAW
        """
        Make a GPAW calculator
        """
        import gpaw,gpaw.poisson #type: ignore
        from gpaw.utilities import h2gpts #type: ignore
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
    def _mk_vasp(self
                ,atoms_path
                ) : # ignore return type
        """
        Make a VASP calculator
        """
        import ase.calculators.vasp as vasp_calculator #type: ignore

        if self.quality == 'tom': xc = 'beef-vdw'
        else:                     xc = 'PBE'

        return vasp_calculator.Vasp(encut = pw_dict[self.quality] ,xc = xc
                                    ,kpts = self._mk_kpts(atoms_path),npar = 1
                                    ,algo= 'fast',prec = 'accurate',nsw =  0
                                    ,nelmdl=3 ,isym= 0 ,lcharg = True,laechg = True)

    def _mk_qe(self
              ,atoms_path
              ) : # ignore return type
        """
        Make a QE calculator
        """
        from espresso import espresso #type: ignore
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

    def _sort_dict(self
                  ,analysis_path
                  ) :
        """
        Read ase-sort.dat. Only should be called if VASP
        """

        with open(join(analysis_path,'ase-sort.dat'),'r') as f:
            loglines = map(lambda x: map(int,x.split()),f.readlines())
            return {x:y for x,y in loglines}

    def _core_dict(self
                  ,analysis_path
                  ):
        """
        For QE calculations, get {atomic number : # core electrons}
        """
        cordict = {}
        with open(join(analysis_path,'calcdir','log'),'r') as f:
            loglines = list(filter(None,[x.split() for x in f.readlines()]))
            for i,line in enumerate(loglines):
                if line[0] == 'PseudoPot.':                      # for each PSP:
                    elem = line[4]                               # symbol
                    num  = ase.data.chemical_symbols.index(elem) # atomic number
                    val  = int(float((loglines[i+3][-1])))       # valence electrons
                    cordict[num] = num - val                     # core electrons
        return cordict

    # EXPOSED METHODS
    #-----------------
    def submit(self
              ,working_path
              ,trajname
              ) :
        """
        Hlepful docstring
        """
        analysis_path,trajpth= self._create_analysis_folder(working_path,trajname)
        checks = [analysis_path in running_jobs_sherlock()
                 ,exists(join(analysis_path,'bonds.json'))]
        if not any(checks):
            self._write_metascript(analysis_path)
            self._write_script(analysis_path,working_path,trajname)
            os.chdir(analysis_path)
            os.system('sbatch %s'%join(analysis_path,'sub_chargemol.sh'))

    def analyze(self
               ,working_path
               ,trajname
               ,start_ind     = 0
               ) :
        """
        Start the process at any point using start_ind
        """

        analysis_path,trajpth= self._create_analysis_folder(working_path,trajname)

        pipeline = [self._generate_charge_density # 0
                   ,self._write_chargemol_input   # 1
                   ,self._call_chargemol          # 2
                   ,self._postprocess             # 3
                   ,self._write_metadata]         # 4

        for process in pipeline[start_ind:]:
            process(analysis_path,trajpth)

    def launch_neb(self
                  ,neb_path
                  ) :
        """
        Take a neb folder to generate chargemol jobs for all images
        """
        os.chdir(neb_path)
        for neb in glob.glob('neb?.traj'): self.submit(neb_path,neb)

################################################################################
################################################################################
################################################################################

####################
# Parsing Classes
#--------------------

class PotentialEdge(object):
    """
    Container for information we need to decide later if it's a
    graph-worthy bond. This class is nothing more than a dictionary.
    """
    def __init__(self
                ,fromNode
                ,toNode
                ,distance
                ,offset
                ,bond_order
                ,tot
                ) :

        self.fromNode = fromNode
        self.toNode   = toNode
        self.distance = round(distance,2)
        self.offset   = offset
        self.bondorder= round(bond_order,3)
        self.total_bo = round(tot,2)


class BondOrderSection(object):
    """Process one section of the Bond Order output of Chargemol"""
    def __init__(self
                ,atoms
                ,ind
                ,sumBO
                ,raw_lines
                ,sortdict
                ,pbcdict
                ) :

        self.atoms    = atoms
        self.ind      = ind
        self.sumBO    = sumBO
        self.bonds    = [parse_line(x) for x in raw_lines]
        self.sortdict = sortdict
        self.pbcdict  = pbcdict

    def _relative_shift(self
                       ,i
                       ,j
                       ) :
        """
        Given a pbc_dict and two indices, return the original pbc shift
        for a bond from i to j
        """
        pi,pj = [np.array(self.pbcdict[x]) for x in [i,j]]
        return pj - pi

    def makeEdge(self
                ,tup
                ) :
        """
        Creates an Edge instance from the result of a parsed Bond Order log line
        """
        (toInd,bo,offset) = tup
        fromInd = self.ind

        if self.sortdict: # undo VASP reordering if necessary
            fromInd,toInd = [self.sortdict[x] for x in [fromInd,toInd]] # type: ignore

        # correct for WRAPPING atoms
        offset = (np.array(offset) +  self._relative_shift(fromInd,toInd)).tolist()

        shift = np.dot(offset, self.atoms.get_cell()) # PBC shift
        p1    = self.atoms[fromInd].position
        p2    = self.atoms[toInd].position + shift
        d     = np.linalg.norm(p2-p1)

        return PotentialEdge(fromInd,toInd,d,offset,bo,self.sumBO)

    def make_edges(self):
        """Apply edgemaker to result of parsing logfile lines"""
        return  [self.makeEdge(b) for b in self.bonds]


###################
# Parsing Functions
#-------------------

def parse_chargemol(analysis_path
                   ,atoms_path
                   ,sortdict
                   ):
    """
    Read file DDEC6_even_tempered_bond_orders.xyz
    """
    header,content    = [], []
    sections          = []
    head_flag,counter = True,-1       # Initialize
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

def parse_line(line ) :
    """
    Get bonded atom, bond order, and offset
    """
    assert line[:16]==' Bonded to the (', "No parse line ->"+line+'<- %d'%len(line)
    offStr,indStr,boStr = line[16:].split(')')
    offset = [int(x) for x in offStr.split(',')] # Chunk containing offset info
    ind    = int(indStr.split()[-3]) - 1        # Chunk containing index info (chargemol starts at 1, not 0)
    bo     = float(boStr.split()[4])            # Chunk containing B.O. info
    return (ind,bo,offset)

####
# Misc
#-----

def parse_chargemol_pbc(header_lines
                       ,cell
                       ) :
    """
    Helpful docstring
    """
    atoms = ase.Atoms(cell=cell)
    for i,l in enumerate(header_lines[2:]):
        try:
            s,x,y,z,_ = l.split()
            p = [float(q) for q in [x,y,z]]
            print(p)
            atoms.append(ase.Atom(s,position=p))
        except Exception as e: print(e)

    return mk_pbc_dict(atoms)


def mk_pbc_dict(atoms
               ) :
    """
    Helpful docstring
    """
    def g(tup ):
        """
        Helper function to yield tuples for pbc_dict
        """

        def f(x):
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

def dict_diff(d1
             ,d2
             ):
    """
    Helpful docstring
    """
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
