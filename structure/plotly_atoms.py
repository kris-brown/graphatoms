# External Modules
import plotly.plotly     as py
import plotly.graph_objs as go
from plotly.plotly import plot as onlineplot
from plotly.offline import plot as off_plot
from plotly.grid_objs import Grid, Column
import time

import itertools,os,ase,ase.io,sys
from copy import deepcopy
import numpy as np
from ase.data import covalent_radii, atomic_numbers
from CataLog.misc.utilities import merge_dicts,negate, flatten
from CataLog.structure.graph import GraphMaker


user = os.environ['USER']
#####
class PlotlyAtoms(object):
    """
    Object for creating atoms objects and theircorresponding network graphs
    """

    def __init__(self, graph,show_indices=None,repeat = (1,1)):
        #Assertions for input data
        """
        IN FUTURE, STORE CELL AS ATTRIBUTE TO GRAPH, ALL OTHER INFO IN NODES
            SO THAT WE DONT HAVE TO PASS AN *ATOMS* OBJECT HERE
        """

        #Set member data
        self.graph                  = graph
        self.cell                   = self.graph.graph['cell']

        self.Xn, self.Yn, self.Zn   = zip(*map(lambda x: x['position'],self.graph.node.values()))

        #Chemical Info
        self.chemical_symbols       = map(lambda x: x['symbol'],self.graph.node.values())
        self.chemical_formula       = ''.join(map(lambda symb: symb+str(self.chemical_symbols.count(symb)),np.sort(list(set(self.chemical_symbols)))))

        #Visualization Parameters
        self.repeat                 = repeat
        if show_indices is None:
            self.show_indices       = range(len(self.Xn))
        else:
            if -1 in show_indices:
                show_indices.append(len(self.Xn)-1)
            self.show_indices       = show_indices

        #Create traces and layout
        self.set_grid_data()
        self.data                   = {}
        self.edge_trace             = self.get_edge_trace()
        self.node_trace             = self.get_node_trace()
        self.data                   = self.node_trace+self.edge_trace
        self.set_layout()


    def set_grid_data(self,size_factor = 40):
        from label import label2Color
        colors = map(label2Color, self.chemical_symbols)
        size = map(lambda symb: covalent_radii[atomic_numbers[symb]]*size_factor, self.chemical_symbols)
        text = self.get_atom_labels()

        for x_shift in range(self.repeat[0])[1:]:
            for y_shift in range(self.repeat[1])[1:]:
                shift = np.dot([x_shift,y_shift,0], self.cell)
                self.Xn += [self.Xn+shift[0]]
                self.Yn += [self.Xn+shift[1]]
                self.Zn += [self.Xn+shift[2]]

        self.Xe,self.Ye,self.Ze               = [],[],[]
        counted_edges,edge_labels,edge_colors = [],[],[]

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
                            edge_label = [str((i,j))]+map(lambda key: str(edge_dict.get(key,'')),['bondorder','weight'])
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

    def get_atom_labels(self):
        atom_indices_label      = ['Index: {0}'.format(x) for x in self.graph.nodes()]
        symbols_label           = ['Symbol: {0}'.format(x) for x in self.chemical_symbols]
        coordination_num_label  = ['Coord #: {0}'.format(x) for x in self.graph.degree().values()]

        text = ['<br>'.join(string) for string in  zip(atom_indices_label,symbols_label,coordination_num_label)]
        return text



    def _get_pbc_shift(self,ind_1,ind_2,count):
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
        axis=dict(showbackground=False,
                  showline=True,
                  zeroline=True,
                  showgrid=True,
                  showticklabels=True,
                  autorange=True
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

    def plot(self, file_name = 'atoms_obj.html', offline = True):
        fig=go.Figure(data=self.data, layout=self.layout)
        if offline:
            output = off_plot(fig, filename = '/home/{0}/scp/plotly/{1}'.format(user,file_name),auto_open = False)
        else:
            output = onlineplot(fig, auto_open = False, sharing = 'public', filename = file_name)
        print output
        return self.graph


def plotly_atoms_animation(graph_array, file_name = 'atoms_obj.html',show_indices = None, repeat = (1,1), offline = True):
        plotly_atoms_list = map(lambda graph: PlotlyAtoms(graph, show_indices, repeat), graph_array)
        layout = plotly_atoms_list[0].layout
        data = plotly_atoms_list[0].data
        frames   = [{'data':plotly_atoms.data,'name':str(i)} for i, plotly_atoms in enumerate(plotly_atoms_list)]
        # data   = flatten(data)


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
                                },
                                {
                                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                    'transition': {'duration': 0}}],
                                    'label': 'Pause',
                                    'method': 'animate'
                                }
                            ],
                            'direction': 'left',
                            'pad': {'r': 10, 't': 87},
                            'showactive': False,
                            'type': 'buttons',
                            'x': 0.1,
                            'xanchor': 'right',
                            'y': 1,
                            'yanchor': 'top'
                        }
                    ]

        fig    = go.Figure(data=data, layout=layout, frames = frames)
        if offline:
            output = off_plot(fig, filename = '/home/{0}/scp/plotly/{1}'.format(user,file_name.strip('.html')))
        else:
            output =  py.create_animations(fig, auto_open = False, sharing = 'public', filename = file_name)
        print output



def plotly_atoms_animation(graph_array, file_name = 'atoms_obj.html',show_indices = None, repeat = (1,1), offline = True):
        plotly_atoms_list = map(lambda graph: PlotlyAtoms(graph, show_indices, repeat), graph_array)
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
        print output

def plotly_neb(pth,num_of_images=7,show_indices = [-1]):
    graph_array = map(lambda ind: GraphMaker(include_frac=0.9,group_cut=0.3,min_bo=0.03).make_graph(pth,'neb{}'.format(ind)),range(num_of_images))
    plotly_atoms_animation(graph_array,show_indices = show_indices, offline = False, file_name = None)

if __name__ == '__main__':
    rootdir = '/scratch/users/ksb/demos/mike_demo/'
    pth = os.path.join(rootdir,sys.argv[1])
    plotly_neb(pth)

    # def slider_plot(self, plateau_array = np.arange(0.05,0.3,0.05), file_name = 'atoms_obj.html', cell = False, offline = True):
    #     from plotly.plotly import plot as onlineplot
    #     from plotly.offline import plot as off_plot
    #     from plotly.offline import iplot as off_iplot
    #
    #
    #     edge_array = [self.get_edge_trace() for plateau_temp in plateau_array]
    #     layout = deepcopy(self.layout)
    #
    #     #Make the Slider
    #     steps = []
    #     for i in range(len(plateau_array)):
    #         step = dict(
    #             method = 'restyle',
    #             label = str(plateau_array[i]),
    #             args = ['visible', [False] * (len(plateau_array)+1)],
    #         )
    #         step['args'][1][i] = True # Toggle i'th trace to "visible"
    #         step['args'][1][-1] = True
    #         steps.append(step)
    #
    #     sliders = [dict(
    #                     active = np.floor(len(plateau_array))/2,
    #                     currentvalue = {"prefix": "Neighbor Plateau: "},
    #                     pad = {"t": 50},
    #                     steps = steps
    #                 )]
    #     layout.update({'sliders':sliders})
    #
    #     trace = edge_array+[self.data['node_trace']]
    #     fig=go.Figure(data=trace, layout=layout)
    #     if offline:
    #         output = off_plot(fig, filename = '/home/{0}/scp/plotly/{1}'.format(user,file_name),auto_open = False)
    #     else:
    #         output = onlineplot(fig, auto_open = False, sharing = 'public', filename = file_name)
    #     print output
    #
    #
    #
    # def get_cell_trace(self):
    #     x = [0, 0, 1, 1, 0, 0, 1, 1]
    #     y = [0, 1, 1, 0, 0, 1, 1, 0]
    #     z = [0, 0, 0, 0, 1, 1, 1, 1]
    #     vecs = np.array(zip(x,y,z))
    #     cell_vertices = np.dot(vecs, self.atoms.cell)
    #     cell_trace = go.Mesh3d(
    #                              x = cell_vertices[:,0]
    #                             ,y = cell_vertices[:,1]
    #                             ,z = cell_vertices[:,2]
    #                             ,i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    #                             ,j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    #                             ,k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    #                             ,opacity = 1
    #                             ,vertexcolor = 1
    #                             ,facecolor= 1
    #                             )
    #     return cell_trace