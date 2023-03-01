from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from graphUtils import add_edges, add_random_food
import torch

class Visualizer():
    def __init__(self, wrap_around):
        self.figure = None
        self.graph = None
        canvas_scale = 1
        self.borders = canvas_scale * np.array([-1, -1, 1, 1])  # Hard borders of canvas
        self.scatter_cell = None
        self.scatter_food = None
        self.edge_plot = None
        self.axes = None
        self.device = torch.device('cpu')
        self.wrap_around = wrap_around

    def plot_organism(self, graph):
        any_edges = add_edges(graph, 0.05, self.device, self.wrap_around, batch_size=1)
        cellIndices = torch.nonzero(graph.x[:, 4] == 1).flatten()
        foodIndices = torch.nonzero(graph.x[:, 4] == 0).flatten()

        edge_from = graph.edge_index[0, :]
        edge_to = graph.edge_index[1, :]
        node_from = graph.x[edge_from, :2].detach().cpu().numpy()
        node_to = graph.x[edge_to, :2].detach().cpu().numpy()
        edges_x = [node_from[:,0], node_to[:,0]]
        edges_y = [node_from[:,1], node_to[:,1]]

        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            self.axes = plt.axes(xlim=self.borders[::2], ylim=self.borders[1::2])
            self.scatter_cell = self.axes.scatter(
                graph.x[cellIndices, 0],
                graph.x[cellIndices, 1],
                marker=".",
                edgecolor="k",
                lw=0.5,
                #**kwargs
            )
            self.scatter_food = self.axes.scatter(
                graph.x[foodIndices, 0],
                graph.x[foodIndices, 1],
                marker=".",
                edgecolor="r",
                #lw=0.5,
                s=graph.x[foodIndices, 2]*3,
                #**kwargs
            )
            self.edge_plot = self.axes.plot([[],[]], [[],[]], linewidth=0.1)
            plt.show()

        self.scatter_cell.set_offsets(graph.x[cellIndices, :2])
        self.scatter_food.set_offsets(graph.x[foodIndices, :2])
        [plot.remove() for plot in self.edge_plot]
        self.edge_plot = self.axes.plot(edges_x, edges_y, linewidth=0.1)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def animate_organism(self, graph, model, food=100, frames=50, interval=150):
        self.graph = graph
        self.plot_organism(graph.clone().detach().cpu())
        #add_random_food(graph, self.device, food)

        @torch.no_grad()
        def animate(i):
            self.graph = model.update(self.graph)
            #self.plot_organism(self.graph.clone().detach().cpu())
            self.plot_organism(self.graph.detach().cpu())

        anim = animation.FuncAnimation(self.figure, animate, frames=frames, interval=interval)
        return anim

    def save_animation_to_gif(self, anim, name, fps=30):
        anim.save('../animation/' + name + '.gif', writer='imagemagick', fps=fps)