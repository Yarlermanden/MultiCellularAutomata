from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

import torch

class Visualizer():
    def __init__(self):
        self.figure = None
        self.graph = None
        canvas_scale = 1
        self.borders = canvas_scale * np.array([-1, -1, 1, 1])  # Hard borders of canvas
        self.scatter = None

    def plot_organism(self, graph):
        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            axes = plt.axes(xlim=self.borders[::2], ylim=self.borders[1::2])
            self.scatter = axes.scatter(
                graph.x[:, 0],
                graph.x[:, 1],
                marker=".",
                edgecolor="k",
                lw=0.5,
                #**kwargs
            )
            # anim = animation.FuncAnimation(figure, animate, frames=50, interval=1)
            plt.show()
        self.scatter.set_offsets(graph.x[:, :2])
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def animate_organism(self, graph, model, frames=50, interval=150):
        self.graph = graph
        self.plot_organism(graph.clone().detach().cpu())

        @torch.no_grad()
        def animate(i):
            self.graph, _, _ = model(self.graph,1)
            self.plot_organism(self.graph.clone().detach().cpu())

        anim = animation.FuncAnimation(self.figure, animate, frames=frames, interval=interval).to_jshtml()
        return anim