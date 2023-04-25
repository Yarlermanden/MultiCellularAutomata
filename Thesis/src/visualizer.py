from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import math
import random
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from datastructure import DataStructure
from graphUtils import add_random_food, cell_mask, food_mask, wall_mask

class Visualizer():
    def __init__(self, settings):
        self.settings = settings
        self.wrap_around = settings.wrap_around
        self.batch_size = settings.batch_size
        self.scale = settings.scale
        self.figure = None
        self.graph = None
        self.scatter_cell = None
        self.scatter_food = None
        self.scatter_wall = None
        self.edge_plot = None
        self.axes = None
        self.texts = None
        #point size = ppi of 72
        #figure ppi defaults to 100
        #1 point == fig.dpi/72. pixels
        self.wall_size = (400*self.settings.radius_wall_damage)**2/(self.scale**2)
        self.cell_size = 80/(self.scale**2)
        self.borders = self.scale * np.array([-1, -1, 1, 1])  # Hard borders of canvas
        self.device = torch.device('cpu')
        self.rows = 2
        self.columns = math.ceil(self.batch_size / 2)
        if self.batch_size < 4:
            self.rows = 1
            self.columns = self.batch_size
        self.datastructure = DataStructure(settings)

    def plot_organism(self, graph):
        any_edges = self.datastructure.add_edges(graph)

        if self.figure is None:
            plt.ion()
            self.figure, self.axes = plt.subplots(self.rows, self.columns, figsize=(10,5))
            self.axes = self.axes.reshape(-1)
            self.figure.set_size_inches(20, 10, True)
            [ax.set_xlim(self.borders[::2]) for ax in self.axes]
            [ax.set_ylim(self.borders[1::2]) for ax in self.axes]
            self.scatter_cell = [ax.scatter(
                [],
                [],
                marker=".",
                edgecolor="k",
                lw=0.5/self.scale,
            ) for ax in self.axes]
            self.scatter_food = [ax.scatter(
                [],
                [],
                marker=".",
                edgecolor="r",
            ) for ax in self.axes]
            self.scatter_wall = [ax.scatter(
                [],
                [],
                marker=".",
                edgecolor="g",
            ) for ax in self.axes]
            self.edge_plot = [ax.plot([[],[]], [[],[]], linewidth=0.1/self.scale) for ax in self.axes]
            self.texts = [self.axes[i].text(0.98, 0.98, '', horizontalalignment='right', verticalalignment='top', transform=self.axes[i].transAxes) for i in range(self.batch_size)]
            plt.show()

        s_idx = 0
        for i in range(self.batch_size):
            e_idx = graph.subsize[i] + s_idx
            cellIndices = torch.nonzero(cell_mask(graph.x[s_idx:e_idx])).flatten() + s_idx
            foodIndices = torch.nonzero(food_mask(graph.x[s_idx:e_idx])).flatten() + s_idx
            wallIndices = torch.nonzero(wall_mask(graph.x[s_idx:e_idx])).flatten() + s_idx

            if any_edges:
                [plot.remove() for plot in self.edge_plot[i]]

                edges_in_batch = torch.nonzero(torch.isin(graph.edge_index[1], cellIndices)).view(-1)

                #TODO the wraparound isn't visualized correctly
                edge_from = graph.edge_index[0, edges_in_batch]
                edge_to = graph.edge_index[1, edges_in_batch]
                node_from = graph.x[edge_from, :2].detach().cpu().numpy()
                node_to = graph.x[edge_to, :2].detach().cpu().numpy()
                edges_x = [node_from[:,0], node_to[:,0]]
                edges_y = [node_from[:,1], node_to[:,1]]
            else:
                edges_x = [[]]
                edges_y = [[]]
            #self.edge_plot[i] = self.axes[i//self.columns][i%self.columns].plot(edges_x, edges_y, linewidth=0.1)
            self.edge_plot[i] = self.axes[i].plot(edges_x, edges_y, linewidth=0.1/self.scale)
            #TODO set size of cells depending on energy level
            self.scatter_food[i].set_offsets(graph.x[foodIndices, :2])
            self.scatter_food[i].set_sizes(graph.x[foodIndices, 2]*10/(self.scale**2))
            self.scatter_wall[i].set_offsets(graph.x[wallIndices, :2])
            self.scatter_wall[i].set_sizes([self.wall_size]*len(wallIndices))
            self.scatter_cell[i].set_offsets(graph.x[cellIndices, :2])
            self.scatter_cell[i].set_sizes([self.cell_size]*len(cellIndices))
            #self.axes[i].text(0.98, 0.98, 'Food: ' + str(int(graph.food_reward[i].item())), horizontalalignment='right', verticalalignment='top', transform=self.axes[i].transAxes)
            self.texts[i].set_text('Food: ' + str(int(graph.food_reward[i].item())))
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            s_idx = e_idx

    def animate_organism(self, graph, model, interval=150):
        self.graph = graph
        self.plot_organism(graph.clone().detach().cpu())

        def gen(): #plot until 10 frames after all cells die
            count_down = 10
            while count_down > 0:
                graph = model.update(self.graph)
                self.graph = graph
                if cell_mask(self.graph.x).sum() == 0: #no more cells
                    count_down -= 1
                yield graph

        @torch.no_grad()
        def animate(graph):
            self.plot_organism(graph.detach().cpu())

        anim = animation.FuncAnimation(self.figure, animate, frames=gen, interval=interval, save_count=2000)
        return anim

    def save_animation_to_gif(self, anim, name, fps=30, dpi=200):
        anim.save('../animation/' + name + '.gif', writer='imagemagick', fps=fps, dpi=dpi)

    def plot_graph(self, graph):
        '''
        Plot a graph in a tree-like structure, where the root node is the news article.
        '''
        #any_edges = self.datastructure.add_edges(graph)
        nodes2 = graph.x[:graph.subsize[0]]
        #edges2 = graph.edge_index[graph.subsize[0]]
        #edges2 = graph.edge_index[torch.nonzero(torch.isin(graph.edge_index[0], torch.nonzero(nodes2))).view(-1)]
        edges2 = graph.edge_index[:, torch.nonzero(graph.edge_index[0] < graph.subsize[0])].squeeze()
        graph2 = Data(nodes2, edges2)
        G = to_networkx(graph2)
        N = G.number_of_nodes()

        figure, axes = plt.subplots(1, 1, figsize=(10,5))

        nodes = G.nodes()

        #different node types:
        cells = torch.nonzero(cell_mask(graph2.x)).flatten()
        food = torch.nonzero(food_mask(graph2.x)).flatten()
        walls = torch.nonzero(wall_mask(graph2.x)).flatten()
        
        root = cells[0]

        # Define positions to draw hierarchical diagram. See pydoc for source.
        #pos = self.hierarchy_pos(G=G, root=root)
        #pos = nx.spring_layout(G)
        pos = nx.shell_layout(G)

        # Define the colors
        node_color = [None] * N
        for cell_idx in cells:
            node_color[cell_idx] = 'tab:blue'
        for food_idx in food:
            node_color[food_idx] = 'tab:red'
        for wall_idx in walls:
            node_color[wall_idx] = 'tab:green'

        # Define node sizes
        node_sizes = [1e3 // N] * N
        node_sizes[root] = node_sizes[0]*2

        #nx.draw_networkx_nodes
        nx.draw(
        #nx.draw_networkx_nodes(
        #nx.draw_networkx(
            G=G, 
            pos=pos, 
            node_color=node_color, 
            #labels={root: 'Root'},
            edge_color= 'tab:gray',
            node_size=node_sizes,
            width=[0.3],
            ax=axes
        )