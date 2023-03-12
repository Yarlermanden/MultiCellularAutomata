import grispy as gsp
import numpy as np
import torch
import time

class FixedRadiusNearestNeighbors(object):
    def __init__(self, nodes):
        blocks = 10 # 10x10
        #grid = gsp.GriSPy(nodes, N_cells=blocks)
        Lbox = 1.0
        periodic = {0: (-Lbox, Lbox), 1: (-Lbox, Lbox), 2: None}
        #grid = gsp.GriSPy(nodes, periodic=0.2)

        self.grid = gsp.GriSPy(nodes.detach().cpu().numpy(), periodic=periodic, N_cells=blocks)
        #self.grid = gsp.GriSPy(nodes.detach().cpu().numpy(), N_cells=blocks)

    def get_neighbors(self, node, radius):
        bubble_dist, bubble_ind = self.grid.bubble_neighbors(
            node.detach().cpu().numpy(), distance_upper_bound=radius
        )
        return bubble_dist, bubble_ind


class DataStructure(object):
    def __init__(self, radius, device, wrap_around, batch_size):
        self.radius = radius
        self.device = device
        self.wrap_around = wrap_around
        self.batch_size = batch_size
        self.radius_food = radius*5

    def add_edges(self, graph):
        edges = []
        edge_attributes = []

        def add_edge(i: int, j: int, wrap_around: bool, with_food: bool):
            radius_to_use = self.radius
            cell_to_cell = 1
            if with_food:
                radius_to_use = self.radius_food
                cell_to_cell = 0

            xy_dist = (graph.x[i]-graph.x[j])[:2]
            if wrap_around:
                if xy_dist[0] > 1.0:
                    xy_dist[0] = -2.0 + xy_dist[0]
                elif xy_dist[0] < -1.0:
                    xy_dist[0] = 2.0 + xy_dist[0]
                if xy_dist[1] > 1.0:
                    xy_dist[1] = -2.0 + xy_dist[1]
                elif xy_dist[1] < -1.0:
                    xy_dist[1] = 2.0 + xy_dist[1]
            dist = xy_dist.norm()
            if dist <= radius_to_use:
                edges.append([i, j])
                edge_attribute1 = [dist, xy_dist[0], xy_dist[1], cell_to_cell]
                edge_attributes.append(edge_attribute1)

                edges.append([j, i])
                edge_attribute2 = [dist, -xy_dist[0], -xy_dist[1], cell_to_cell]
                edge_attributes.append(edge_attribute2)
            #else its cells...
            else:
                print('something is wrong')

        #batch_index = torch.zeros(graph.x.shape[0])
        #prev_idx = 0 
        #for i in range(self.batch_size):
        #    batch_index[prev_idx:prev_idx+graph.subsize[i]] = i+1
        #    prev_idx += graph.subsize[i]

        #nodes = torch.concat((graph.x[:, :2], batch_index.view(-1, 1)), dim=1)
        #nodes = graph.x[:, :2]
        #frnn = FixedRadiusNearestNeighbors(nodes)
        #cell_indices = torch.nonzero(graph.x[:, 4] == 1).flatten()
        #cells = nodes[cell_indices]
        #dists, indices = frnn.get_neighbors(cells, self.radius_food)
        #for ii, i in enumerate(cell_indices):
        #    for j in indices[ii]:
        #        if batch_index[i]==batch_index[j] and i != j:
        #            add_edge(i, j, self.wrap_around)

        s_idx = 0
        for i in range(self.batch_size):
            time1 = time.perf_counter()
            e_idx = s_idx + graph.subsize[i].detach().cpu().numpy()
            nodes = graph.x[s_idx:e_idx, :2]
            frnn = FixedRadiusNearestNeighbors(nodes)
            cell_indices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 1).flatten()
            cells = nodes[cell_indices]
            if len(cells) != 0:
                time2 = time.perf_counter()
                dists, indices = frnn.get_neighbors(cells, self.radius_food)
                for ii, i in enumerate(cell_indices): #for each cell
                    indices[ii] += s_idx
                    for jj, j in enumerate(indices[ii]): #for each neighbor below food radius
                        if i != j: #not itself
                            if graph.x[j, 4] == 1: #is cell
                                if dists[ii][jj] <= self.radius: #is below cell radius
                                    add_edge(i+s_idx, j, self.wrap_around, False)
                            else:
                                add_edge(i+s_idx, j, self.wrap_around, True)
                time3 = time.perf_counter()
            s_idx = e_idx
            #print('find: ', time2-time1)
            #print('add: ', time3-time2)

        if len(edges) == 0:
            graph.edge_index = torch.tensor([[]], dtype=torch.long, device=self.device)
            graph.edge_attr = torch.tensor([[]], dtype=torch.float, device=self.device)
            return False
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).T
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float, device=self.device)
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr
        return True