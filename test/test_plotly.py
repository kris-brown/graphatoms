# External Modules
from typing import Tuple
from ase.visualize import view # type: ignore
from ase.io import read # type: ignore
import os

# Internal Modules
import graphatoms.structure.plotly_atoms   as mp
import graphatoms.misc.atoms               as ma
from graphatoms.misc.utilities         import true
from graphatoms.structure.graph        import GraphMaker,GraphInput
from graphatoms.structure.graph_utils  import graph_summary

################################################################################
grapher = GraphMaker()
make_init_graph = lambda pth: grapher.make_graph(GraphInput(pth,'init'))
##################
# TRAJS TO TEST ON
###################
example     = '/scratch/users/ksb/share/analysis_sherlock/ksb/151034096235/' # 1x1x6
water_ex    = '/scratch/users/ksb/demos/' #trajname = water
mike_examples = ['/scratch/users/ksb/share/analysis_sherlock/mstatt/151056689066/' #NNH3
                ,'/scratch/users/ksb/share/analysis_sherlock/mstatt/151056482997/' # 2 N2s formed
                ,'/scratch/users/ksb/share/analysis_sherlock/mstatt/151056084723/' # ?
                ,'/scratch/users/ksb/share/analysis_sherlock/mstatt/151028685446/' # ?
                ]

def test_plot(stordir      = example
             ,trajname     = 'final'
             ,include_frac = 0.8
             ,jmol         = False
             ,show_groups  = True
             ,filt         = true):

    input = GraphInput(stordir,trajname)

    GraphMaker(include_frac=include_frac
              ,jmol=jmol).plot_bond_order_analysis(input
                                                  ,show_groups = show_groups
                                                  ,filt        = filt)

def mk_plotly(stordir      : str            = example
             ,trajname     : str            = 'final'
             ,min_bo       : float          = 0.03
             ,include_frac : float          = 0.8
             ,repeat       : Tuple[int,int] = (1,1)
             ):

    input = GraphInput(stordir,trajname)

    grph  = GraphMaker(include_frac = include_frac
                      ,min_bo       = min_bo
                      ).make_graph(input)
    mp.PlotlyAtoms(grph,repeat=repeat).plot(offline=False)
