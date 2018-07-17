# External
import typing as typ
import os,math,time,json,sys,io,shutil,random,glob,string,itertools
from os.path    import exists,join,getctime

import ase                          #type: ignore
import numpy as np                  #type: ignore
from ase.units  import Bohr         #type: ignore
from ase.io     import write,read   #type: ignore

# Internal
from graphatoms.misc.utilities import (flatten,safeMkDir,safeMkDirForce,running_jobs_sherlock) #type: ignore
from graphatoms.misc.print_parse  import print_time #type: ignore

"""
FUNCTIONS NEEDED TO GENERATE BONDS.JSON
"""

############
# CONSTANTS
###########
user         = os.environ['USER']
home         = os.environ['HOME']
graphatoms   = os.environ['GRAPHATOMS_PATH']
queue        = 'iric' # 'iric' if os.environ['SHERLOCK'] == '1' else 'normal'


conv_dict = {'low':1,       'mid':0.1,     'high': 0.01, 'tom': 0.01}
pw_dict   = {'low':300,     'mid':400,     'high': 500,  'tom': 600}
time_dict = {'low':1,       'mid':2,       'high': 3,    'tom': 4}

logfile   = {'gpaw':'log','vasp':'OUTCAR'}
time_dict2= {'gpaw':0.5, 'vasp':1}

codes     = ['gpaw','vasp']
qualities = ['low','mid','high']
############################################

def test_suite(working_path : str
              ,trajname     : str
              ) -> None:
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
    dftcode - {'gpaw','vasp'}
    quality - {'low','mid','high'}
    """

    def __init__(self
                ,dftcode : str ='gpaw'
                ,quality : str = 'low'
                ) -> None:

        assert dftcode in ['gpaw','vasp']
        assert quality in ['low','mid','high']

        #assert 'VASP_VDW_KERNEL' in os.environ # IF we were to use VASP BEEF functional

        self.dftcode = dftcode
        self.quality = quality

    # PRIVATE METHODS
    #--------------------
    def _create_analysis_folder(self
                               ,root_folder : str
                               ,trajname    : str
                               ) -> typ.Tuple[str,str]:
        """
        Creates a directory for job analysis as well as:
        <root_folder>/chargemol_analysis/<trajname>/
        Returns the analysis folder and traj path
        """
        analysis_folder = join(root_folder,'chargemol_analysis',trajname)
        traj_path       = join(root_folder,'%s.traj'%trajname)

        safeMkDirForce(analysis_folder)
        shutil.copyfile(traj_path,join(analysis_folder,'%s.traj'%trajname))

        #shutil.copyfile(vdwpth,join(analysis_folder,'vdw_kernel.bindat')) # IF VASP BEEF

        return analysis_folder,traj_path

    def _write_metascript(self
                         ,analysis_path : str
                         ) -> None:
        """
        Write a script which, when sbatch'd (ON SHERLOCK), will perform bond order analysis
        """

        content_dict = {'vasp':['source %s/.env/bin/activate'%graphatoms
                               ,'python3 sub_chargemol.py']
                       ,'gpaw':["NTASKS=`echo $SLURM_TASKS_PER_NODE|tr '(' ' '|awk '{print $1}'`"
                               ,"NNODES=`scontrol show hostnames $SLURM_JOB_NODELIST|wc -l`"
                               ,'NCPU=`echo " $NTASKS * $NNODES " | bc`'
                               ,'source /scratch/users/ksb/gpaw/paths.bash'
                               ,'source %s/.env/bin/activate'%graphatoms
                               ,'mpirun -n $NCPU gpaw-python sub_chargemol.py']}

        hours  = time_dict[self.quality] * time_dict2[self.dftcode]

        script = '\n'.join(['#!/bin/bash'
                        ,'#SBATCH -p %s'%queue
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
                     ,analysis_path : str
                     ,working_path  : str
                     ,trajname      : str
                     ) -> None:
        """
        Helpful docstring
        """
        if_gpaw = '_untyped' # if self.dftcode=='gpaw' else ''
        header  = '#!/home/users/vossj/suncat/bin/python_w' if self.dftcode=='vasp' else ''

        script = '\n'.join([header
                           ,'from graphatoms.chargemol.chargemol%s import BondAnalyzer'%if_gpaw
                           ,"BondAnalyzer('%s','%s').analyze('%s','%s')"%(self.dftcode,self.quality,working_path,trajname)
                           ])

        with open(join(analysis_path,'sub_chargemol.py'),'w') as f:
            f.write(script)

    # MAIN PIPELINE METHODS
    #----------------------

    def _generate_charge_density(self
                                ,analysis_path : str
                                ,atoms_path    : str
                                ) -> None:
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
        calc_dict = {'vasp':self._mk_vasp,'gpaw':self._mk_gpaw}
        calc      = calc_dict[self.dftcode](atoms_path)
        atoms.set_calculator(calc)
        atoms.get_potential_energy()

        # Write charge density files if necessary
        #-----------------------------------------
        if self.dftcode == 'gpaw':
            density = calc.get_all_electron_density() * Bohr**3
            write(join(analysis_path,'total_density.cube')
                 ,atoms,data=density)

    def _write_chargemol_input(self
                              ,analysis_path : str
                              ,atoms_path    : str
                              ) -> None:
        """
        Write job_control.txt to an analysis folder
        """

        job_control = '\n'.join(['<net charge>',"0.0","</net charge>"
                                ,"<periodicity along A, B, and C vectors>"
                                ,".true.",".true.",".true."
                                ,"</periodicity along A, B, and C vectors>"
                                ,'<compute BOs>','.true.','</compute BOs>'
                                ,'<atomic densities directory complete path>'
                                ,'{0}/chargemol/atomic_densities/'.format(graphatoms)
                                ,'</atomic densities directory complete path>'
                                ,'<charge type>','DDEC6','</charge type>'
                                ])

        with open(join(analysis_path,'job_control.txt'),'w') as f:
            f.write(job_control)

    def _call_chargemol(self
                       ,analysis_path : str
                       ,atoms_path    : str
                       ) -> None:
        """
        Call chargemol binary from a specified folder
        """

        os.chdir(analysis_path)
        path_to_chargemol = '/scratch/users/ksb/graphatoms/chargemol/chargemol_binary' # need to compile parallel with relaxed tolerance but getting error
        os.system(path_to_chargemol)
        print('executing: ',path_to_chargemol)
        check = join(analysis_path,'DDEC6_even_tempered_bond_orders.xyz')
        if not exists(check):
            raise ValueError(analysis_path+'/DDEC6_even_tempered_bond_orders.xyz should exist')

    # DFTCODE-SPECIFIC METHODS
    #-------------------------
    def _mk_kpts(self
                ,atoms_path : str
                ) -> typ.List[int]:
        """
        Chooses appropriate kpt spacing
        """

        def get_kpt(x : np.array) -> int:
            return int(math.ceil(15/np.linalg.norm(x)))

        cell = read(atoms_path).get_cell() # 3 x 3 array

        return  [get_kpt(v) for v in cell]

    def _mk_gpaw(self
                ,atoms_path : str
                ) -> typ.Any: # ignore return type ... not everyone can import GPAW
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
                ,atoms_path : str
                ) -> typ.Any: # ignore return type
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

    # EXPOSED METHODS
    #-----------------
    def submit(self
              ,working_path : str
              ,trajname     : str
              ) -> None:
        """
        Start a chargemol job as a batch job
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
               ,working_path : str
               ,trajname     : str
               ,start_ind    : int = 0
               ) -> None:
        """
        Start the process at any point using start_ind
        """

        analysis_path,trajpth= self._create_analysis_folder(working_path,trajname)

        pipeline = [self._generate_charge_density # 0
                   ,self._write_chargemol_input   # 1
                   ,self._call_chargemol]         # 2

        for process in pipeline[start_ind:]:
            process(analysis_path,trajpth)
