# External Modules
from typing import Tuple,List
import plotly.plotly     as py #type: ignore
import plotly.graph_objs as go #type: ignore
from plotly.plotly import plot as onlineplot
from plotly.offline import plot as off_plot #type: ignore
from plotly.grid_objs import Grid, Column #type: ignore

import itertools,os,sys,time
import ase,ase.io #type: ignore
from copy import deepcopy
import numpy as np #type: ignore
from ase.data import covalent_radii, atomic_numbers #type: ignore
import networkx as nx #type: ignore

# Internal Modules
from graphatoms.misc.utilities import merge_dicts,negate, flatten
from graphatoms.structure.graph import GraphMaker,GraphInput

"""
Construct Plotly objects from a graph
"""

################################################################################
# Constnats
#-----------
user = os.environ['USER']

#####
labelDict = {
    'H': 'black'      ,'Li': 'purple'       ,'Be':'mediumaquamarine'
    ,'B':'pink'       ,'C':'grey'           ,'N':'blue'
    ,'O':'red'        ,'F':'forestgreen'    ,'Na':'purple'
    ,'Mg':'lightcoral','Al':'firebrick'     ,'Si':'palevioletred'
    ,'P':'orange'     ,'S':'red'            ,'Cl':'green'
    ,'K':'purple'     ,'Ca':'lightsalmon'   ,'Sc':'grey'
    ,'Ti':'grey'      ,'V':'blue'           ,'Cr':'cyan'
    ,'Mn':'purple'    ,'Fe':'darkred'       ,'Co':'pink'
    ,'Ni':'green'     ,'Cu':'brown'         ,'Zn':'indigo'
    ,'Ga':'pink'      ,'Ge':'lightblue'     ,'As':'fuchsia'
    ,'Se':'turquoise' ,'Br':'azure'         ,'Rb':'black'
    ,'Sr':'olive'     ,'Y':'plum'           ,'Zr':'palevioletred'
    ,'Nb':'aqua'      ,'Mo':'khaki'         ,'Tc':'green'
    ,'Ru':'lime'      ,'Rh':'teal'          ,'Pd':'grey'
    ,'Ag':'silver'    ,'Cd':'purple'        ,'In':'blue'
    ,'Sn':'green'     ,'Sb':'red'           ,'Te':'plum'
    ,'I':'red'        ,'Cs':'orange'        ,'Ba':'tan'
    ,'Os':'pink'      ,'Ir':'green'         ,'Pt':'blue'
    ,'Au':'gold'      ,'Pb':'brown'}
#########################################################
class PlotlyAtoms(object):
    """
    Object for creating atoms objects and theircorresponding network graphs
    """

    def __init__(self
                , graph         : nx.Graph
                , show_indices  : List[int]      = None
                ,repeat         : Tuple[int,int] = (1,1)
                ) -> None:
        #Assertions for input data
        """
        IN FUTURE, STORE CELL AS ATTRIBUTE TO GRAPH, ALL OTHER INFO IN NODES
            SO THAT WE DONT HAVE TO PASS AN *ATOMS* OBJECT HERE
        """

        #Set member data
        self.graph                  = graph
        self.cell                   = self.graph.graph['cell']

        posTuples = zip(*[x['position'] for x in self.graph.node.values()])
        self.Xn, self.Yn, self.Zn   = [list(x) for x in posTuples] # convert tuple to list

        #Chemical Info
        self.atomic_numbers         = [x['number'] for x in self.graph.node.values()]
        self.chemical_symbols       = [x['symbol'] for x in self.graph.node.values()]
        self.chemical_formula       = ''.join([symb+str(self.chemical_symbols.count(symb))
                                                  for symb in np.sort(list(set(self.chemical_symbols)))])

        #Visualization Parameters
        self.repeat                 = repeat
        if show_indices is None:
            self.show_indices       = list(range(len(self.Xn)))
        else:
            if -1 in show_indices:
                show_indices.append(len(self.Xn)-1)
            self.show_indices       = show_indices

        #Create traces and layout
        self.set_grid_data()
        self.data                   = {} # type: dict
        self.edge_trace             = self.get_edge_trace()
        self.node_trace             = self.get_node_trace()
        self.data                   = self.node_trace+self.edge_trace
        self.set_layout()


    def set_grid_data(self,size_factor : int = 40) -> None:
        """
        Helpful docstring from mike
        """
        colors = [labelDict[x] for x in self.chemical_symbols]
        size   = [covalent_radii[atomic_numbers[symb]]*size_factor for symb in self.chemical_symbols]
        text   = self.get_atom_labels()

        for x_shift in range(self.repeat[0])[1:]:
            for y_shift in range(self.repeat[1])[1:]:
                shift = np.dot([x_shift,y_shift,0], self.cell)
                self.Xn+=[x + shift[0] for x in self.Xn]
                self.Yn+=[y + shift[1] for y in self.Yn]
                self.Zn+=[z + shift[2] for z in self.Zn]

        self.Xe,self.Ye,self.Ze               = [],[],[] # type: Tuple[list,list,list]
        counted_edges,edge_labels,edge_colors = [],[],[] # type: Tuple[list,list,list]

        for x_shift in range(self.repeat[0]):
            for y_shift in range(self.repeat[1]):
                shift = np.dot([x_shift,y_shift,0], self.cell)
                for (i,j) in list(set(self.graph.edges())):
                    if i in self.show_indices or j in self.show_indices:
                        if i not in self.show_indices:
                            temp = i; i = j; j = temp
                        counts = self.graph.get_edge_data(i,j).keys()
                        for count in counts:
                            edge_dict = self.graph.get_edge_data(i,j)[count]
                            edge_color = self.get_red_blue_color(edge_dict.get('bondorder',None))
                            edge_colors+= [edge_color]*3
                            counted_edges.append((i,j))
                            edge_label = [str((i,j))] + [str(edge_dict.get(key,'')) for key in ['bondorder','weight']]
                            edge_labels += ['edge pair: {0} <br> bond order: {1} <br> distance: {2}'.format(*edge_label)]*3

                            pbc_shift = self._get_pbc_shift(i, j, count)
                            real_dis_shift = np.dot(pbc_shift,self.cell)
                            self.Xe += [self.Xn[i]+shift[0],self.Xn[j]+real_dis_shift[0]+shift[0],None]# x-coordinates of edge ends
                            self.Ye += [self.Yn[i]+shift[1],self.Yn[j]+real_dis_shift[1]+shift[1],None]# y-coordinates of edge ends
                            self.Ze += [self.Zn[i]+shift[2],self.Zn[j]+real_dis_shift[2]+shift[2],None]# z-coordinates of edge ends



        self.grid = Grid([Column(self.Xn,'Xn')])
        self.grid.append(Column(self.Yn,'Yn'))
        self.grid.append(Column(self.Zn,'Zn'))
        self.grid.append(Column(self.Xe,'Xe'))
        self.grid.append(Column(self.Ye,'Ye'))
        self.grid.append(Column(self.Ze,'Ze'))
        self.grid.append(Column(text,'node_text'))
        self.grid.append(Column(size,'node_size'))
        self.grid.append(Column(colors,'node_color'))
        self.grid.append(Column(edge_colors,'edge_colors'))
        self.grid.append(Column(edge_labels,'edge_labels'))
        url = py.grid_ops.upload(self.grid, 'grid_data_plotly_'+str(time.time()), auto_open=False)

    def get_atom_labels(self) -> List[str]:
        atom_indices_label      = ['Index: {0}'.format(x) for x in self.graph.nodes()]
        symbols_label           = ['Symbol: {0}'.format(x) for x in self.chemical_symbols]

        coordination_num_label  = ['Coord #: {0}'.format(x) for _,x in self.graph.degree()]

        text = ['<br>'.join(string) for string in  zip(atom_indices_label,symbols_label,coordination_num_label)]
        return text



    def _get_pbc_shift(self,ind_1 : int ,ind_2 : int ,count : int) -> Tuple[int,int,int]:
        from copy import deepcopy
        pbc_shift = np.array(self.graph[ind_1][ind_2][count]['pbc_shift'])
        if not ind_1 == ind_2:
            dis_best = np.inf
            for x_mult in [-1,0,1]:
                for y_mult in [-1,0,1]:
                    for z_mult in [-1,0,1]:
                        pbc_shift_curr = deepcopy(pbc_shift)
                        pbc_shift_curr[0] = x_mult; pbc_shift_curr[1] = y_mult; pbc_shift_curr[2] = z_mult
                        dis_shift = np.dot(pbc_shift_curr, self.cell)
                        pos_1   = np.array([self.Xn[ind_1],self.Yn[ind_1],self.Zn[ind_1]])
                        pos_2   = np.array([self.Xn[ind_2],self.Yn[ind_2],self.Zn[ind_2]])
                        dis_curr = np.linalg.norm(dis_shift+pos_2-pos_1)
                        if dis_curr < dis_best:
                            dis_best = dis_curr
                            pbc_shift_best = pbc_shift_curr
        else:
            pbc_shift_best = pbc_shift
        return pbc_shift_best

    @staticmethod
    def get_red_blue_color(value):
        """Convert value between 0 and 1 into an rgb color
        0 will be blue; 1 will be red"""
        if not value==None:
            return 'rgb({0},0,{1})'.format(255*value,255*(1-value))
        else:
            return 'rgb(0,0,0)'

    def get_node_trace(self):
        node_trace = go.Scatter3d(mode = 'markers',
                                  xsrc=self.grid.get_column_reference('Xn'),
                                  ysrc=self.grid.get_column_reference('Yn'),
                                  zsrc=self.grid.get_column_reference('Zn'),
                                  name='Nodes',
                                  marker=dict(symbol='dot',
                                             sizesrc=self.grid.get_column_reference('node_size'),
                                             sizeref = 1,
                                             colorsrc=self.grid.get_column_reference('node_color'),
                                             line=go.Line(color='rgb(0,0,0)', width=1),
                                             opacity=1
                                             ),
                                  textsrc=self.grid.get_column_reference('node_text'),
                                  hoverinfo='text'
                                  )
        return [node_trace]

    def get_edge_trace(self):
        edge_trace = go.Scatter3d(
                            xsrc=self.grid.get_column_reference('Xe'),
                            ysrc=self.grid.get_column_reference('Ye'),
                            zsrc=self.grid.get_column_reference('Ze'),
                            mode='lines',
                            line=go.Line(colorsrc=self.grid.get_column_reference('edge_colors'), width=8),
                            textsrc = self.grid.get_column_reference('edge_labels'),
                            hoverinfo =  "text"
                            )
        return [edge_trace]

    def set_layout(self):

        axis=dict(showbackground= False,
                  showline      = True,
                  zeroline      = True,
                  showgrid      = True,
                  showticklabels= True,
                  autorange     = True
                  )
        x_axis = merge_dicts([{'title':'X'},axis])
        y_axis = merge_dicts([{'title':'Y'},axis])
        z_axis = merge_dicts([{'title':'Z'},axis])

        self.layout = go.Layout(title=self.chemical_formula
                        ,dragmode = "turntable"
                        ,width=1000
                        ,height=1000
                        ,showlegend=False
                        ,scene=go.Scene(xaxis=go.XAxis(x_axis)
                                    ,yaxis=go.YAxis(y_axis)
                                    ,zaxis=go.ZAxis(z_axis)
                                    )
                        ,margin=go.Margin(t=100)
                        ,hovermode='closest'
                        )

    def plot(self
            , file_name = 'atoms_obj.html'
            , offline = True):
        fig=go.Figure(data=self.data, layout=self.layout)
        if offline:
            output = off_plot(fig, filename = '/home/{0}/scp/plotly/{1}'.format(user,file_name),auto_open = False)
        else:
            output = onlineplot(fig, auto_open = False, sharing = 'public', filename = file_name)
        print(output)
        return self.graph

def plotly_atoms_animation(graph_array
                          ,file_name    : str            = 'atoms_obj.html'
                          ,show_indices : List[int]      = None
                          ,repeat       : Tuple[int,int] = (1,1)
                          ,offline      : bool           = True
                          ) -> None:

        plotly_atoms_list = [PlotlyAtoms(graph, show_indices, repeat) for graph in graph_array]
        layout = plotly_atoms_list[0].layout
        data = plotly_atoms_list[0].data
        frames   = [{'data':plotly_atoms.data,'name':str(i)} for i, plotly_atoms in enumerate(plotly_atoms_list)]

        #Make the Slider
        steps = []
        for i in range(len(graph_array)):
            steps.append({'args': [
                [i],
                {'frame': {'duration': 300, 'redraw': False},
                 'mode': 'immediate',
               'transition': {'duration': 300}}
             ],
             'label': i,
             'method': 'animate'})

        sliders = [dict(
                        active = 0,
                        transition = {'duration': 300, 'easing': 'cubic-in-out'},
                        currentvalue = {"prefix": "Image Number: "},
                        pad = {"t": 50},
                        steps = steps
                    )]
        layout.update({'sliders':sliders})

        layout['updatemenus']   = [
                        {
                            'buttons': [
                                {
                                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'cubic-in-out'}}],
                                    'label': 'Play',
                                    'method': 'animate'
                                }]}]

        fig    = go.Figure(data=data, layout=layout, frames = frames)
        if offline:
            output = off_plot(fig, filename = '/home/{0}/scp/plotly/{1}'.format(user,file_name.strip('.html')))
        else:
            output =  py.create_animations(fig, auto_open = False, sharing = 'public', filename = file_name)
        print(output)

def plotly_neb(pth           : str
              ,num_of_images : int       = 7
              ,show_indices  : List[int] = [-1]
              ) -> None:

    gm = GraphMaker(include_frac=0.9,group_cut=0.3,min_bo=0.03)

    def f(ind): return gm.make_graph(GraphInput(pth,'neb{}'.format(ind)))

    graph_array = [f(x) for x in range(num_of_images)]

    plotly_atoms_animation(graph_array,show_indices = show_indices, offline = False, file_name = None)





if __name__ == '__main__':
    rootdir = '/scratch/users/ksb/demos/mike_demo/'
    pth = os.path.join(rootdir,sys.argv[1])
    plotly_neb(pth)
