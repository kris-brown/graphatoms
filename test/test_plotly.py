# External Modules
from typing import Tuple
from ase.visualize import view # type: ignore
from ase.io import read # type: ignore
import os

# Internal Modules
from graphatoms.structure.plotly_atoms   import PlotlyAtoms
from graphatoms.misc.utilities         import true
from graphatoms.structure.graph        import GraphMaker,GraphInput
from graphatoms.structure.graph_utils  import graph_summary
from graphatoms.misc.mysql 			   import realDB,sqlselect
from graphatoms.misc.print_parse       import json_to_graph
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

def vis_id(i      : int            = 1
		  ,repeat : Tuple[int,int] = (2,2)
		  ) -> None:
	q = """SELECT geo_graph FROM struct
			JOIN finaltraj USING (struct_id)
			JOIN job USING (job_id)
			WHERE job_id=%s"""
	output = sqlselect(realDB,q,[i])[0][0]

	if output is None:
		print('no final structure for job_id %d'%i)
	else:
		G = json_to_graph(output)
		PlotlyAtoms(G,repeat=repeat).plot(offline=False)

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
			 ,repeat       : Tuple[int,int] = (2,2)
			 ):

	input = GraphInput(stordir,trajname)

	grph  = GraphMaker(include_frac = include_frac
					  ,min_bo       = min_bo
					  ).make_graph(input)
	PlotlyAtoms(grph,repeat=repeat).plot(offline=False)
