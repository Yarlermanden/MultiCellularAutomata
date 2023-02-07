from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from graphUtils import add_edges, add_random_food, consume_food

import torch

class Visualizer():
    def __init__(self):
        self.figure = None
        self.graph = None
        canvas_scale = 1
        self.borders = canvas_scale * np.array([-1, -1, 1, 1])  # Hard borders of canvas
        self.scatter_cell = None
        self.scatter_food = None
        self.edge_plot = None
        self.device = torch.device('cpu')

    def plot_organism(self, graph):
        any_edges = add_edges(graph, 0.05, self.device)
        if not any_edges:
            return
        cellIndices = torch.nonzero(graph.x[:, 4] == 1).flatten()
        foodIndices = torch.nonzero(graph.x[:, 4] == 0).flatten()
        edgeIndices1 = graph.edge_index[0, :]
        edgeIndices2 = graph.edge_index[1, :]
        edges1 = graph.x[edgeIndices1][:, 0:2].T.detach().cpu().numpy()
        edges2 = graph.x[edgeIndices2][:, 0:2].T.detach().cpu().numpy()

        edges_x = [edges1[0,:], edges2[0, :]]
        edges_y = [edges1[1,:], edges2[1, :]]

        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            axes = plt.axes(xlim=self.borders[::2], ylim=self.borders[1::2])
            self.scatter_cell = axes.scatter(
                graph.x[cellIndices, 0],
                graph.x[cellIndices, 1],
                marker=".",
                edgecolor="k",
                lw=0.5,
                #**kwargs
            )
            self.scatter_food = axes.scatter(
                graph.x[foodIndices, 0],
                graph.x[foodIndices, 1],
                marker=".",
                edgecolor="r",
                lw=0.5,
                #**kwargs
            )
            self.edge_plot, *_ = axes.plot(edges_x, edges_y)
            plt.show()

        self.scatter_cell.set_offsets(graph.x[cellIndices, :2])
        self.scatter_food.set_offsets(graph.x[foodIndices, :2])
        self.edge_plot.set_data(edges_x, edges_y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def animate_organism(self, graph, model, frames=50, interval=150):
        self.graph = graph
        self.plot_organism(graph.clone().detach().cpu())
        add_random_food(graph, self.device, 20)

        @torch.no_grad()
        def animate(i):
            self.graph, *_ = model.update(self.graph)
            self.plot_organism(self.graph.clone().detach().cpu())

        anim = animation.FuncAnimation(self.figure, animate, frames=frames, interval=interval)
        return anim

    def save_animation_to_gif(self, anim, name, fps=30):
        anim.save('../animation/' + name + '.gif', writer='imagemagick', fps=fps)