from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from graphUtils import add_random_food
import torch
from datastructure import DataStructure
import math

class Visualizer():
    def __init__(self, wrap_around, batch_size, scale):
        self.figure = None
        self.graph = None
        canvas_scale = scale
        self.borders = canvas_scale * np.array([-1, -1, 1, 1])  # Hard borders of canvas
        self.scatter_cell = None
        self.scatter_food = None
        self.edge_plot = None
        self.axes = None
        self.device = torch.device('cpu')
        self.wrap_around = wrap_around
        self.batch_size = batch_size
        self.rows = 2
        self.columns = math.ceil(self.batch_size / 2)
        if self.batch_size < 4:
            self.rows = 1
            self.columns = self.batch_size
        self.datastructure = DataStructure(0.04, self.device, self.wrap_around, self.batch_size, scale)

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
                lw=0.5,
            ) for ax in self.axes]
            self.scatter_food = [ax.scatter(
                [],
                [],
                marker=".",
                edgecolor="r",
                #lw=0.5,
            ) for ax in self.axes]
            self.edge_plot = [ax.plot([[],[]], [[],[]], linewidth=0.1) for ax in self.axes]
            plt.show()

        s_idx = 0
        for i in range(self.batch_size):
            e_idx = graph.subsize[i] + s_idx
            cellIndices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 1).flatten() + s_idx
            foodIndices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 0).flatten() + s_idx

            self.scatter_cell[i].set_offsets(graph.x[cellIndices, :2])
            self.scatter_food[i].set_offsets(graph.x[foodIndices, :2])
            self.scatter_food[i].set_sizes(graph.x[foodIndices, 2]*5)
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
            self.edge_plot[i] = self.axes[i].plot(edges_x, edges_y, linewidth=0.1)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            s_idx = e_idx

    def animate_organism(self, graph, model, food=100, frames=50, interval=150):
        self.graph = graph
        self.plot_organism(graph.clone().detach().cpu())

        @torch.no_grad()
        def animate(i):
            self.graph = model.update(self.graph)
            self.plot_organism(self.graph.detach().cpu())

        anim = animation.FuncAnimation(self.figure, animate, frames=frames, interval=interval)
        return anim

    def save_animation_to_gif(self, anim, name, fps=30, dpi=200):
        anim.save('../animation/' + name + '.gif', writer='imagemagick', fps=fps, dpi=dpi)