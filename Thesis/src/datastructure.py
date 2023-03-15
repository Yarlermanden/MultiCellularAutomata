import grispy as gsp
import numpy as np
import torch
import time
from typing import Union, Tuple, Optional
from torch import Tensor

class FixedRadiusNearestNeighbors(object):
    def __init__(self, nodes, food_radius):
        Lbox = 1.0
        blocks = int(Lbox/food_radius) # 10x10
        periodic = {0: (-Lbox, Lbox), 1: (-Lbox, Lbox), 2: None}
        self.grid = gsp.GriSPy(nodes.detach().cpu().numpy(), periodic=periodic, N_cells=blocks)

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

        def update_dist1(dist):
            if dist > 1.0:
                return -2.0 + dist
            elif dist < -1.0:
                return 2.0 + dist
            return dist
        vupdate = np.vectorize(update_dist1)

        #@torch.jit.script
        def update_dist(edge_attr1):
            if edge_attr1[1] > 1.0:
                edge_attr1[1] = -2.0 + edge_attr1[1]
            elif edge_attr1[1] < -1.0:
                edge_attr1[1] = 2.0 + edge_attr1[1]
            if edge_attr1[2] > 1.0:
                edge_attr1[2] = -2.0 + edge_attr1[2]
            elif edge_attr1[2] < -1.0:
                edge_attr1[2] = 2.0 + edge_attr1[2]
            return edge_attr1

        def add_edge1(dist, xy_dist, isCell):
            edge_attr1 = [dist, xy_dist[0], xy_dist[1], isCell]
            edges.append([j, i])
            edge_attributes.append(edge_attr1)
            
        x = graph.x.detach().cpu().numpy()
        s_idx = 0
        for i in range(self.batch_size): #TODO could we simply just vectorize this entire thing?
            time1 = time.perf_counter()
            e_idx = s_idx + graph.subsize[i].detach().cpu().numpy()
            nodes = graph.x[s_idx:e_idx, :2]
            if len(nodes) == 0:
                continue
            frnn = FixedRadiusNearestNeighbors(nodes, self.radius_food)
            cell_indices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 1).flatten()
            cells = nodes[cell_indices]
            cell_indices += s_idx
            cell_indices = cell_indices.detach().cpu().numpy()
            if len(cells) != 0:
                dists, indices = frnn.get_neighbors(cells, self.radius_food)
                indices = [x + s_idx for x in indices]
                time2 = time.perf_counter()

                #matrix of whether i == j
                #matrix of xy_dist from x[i]-x[j]
                for ii, i in enumerate(cell_indices): #for each cell
                    for jj, j in enumerate(indices[ii]): #for each neighbor below food radius
                        if i != j: #not itself #TODO could experiment with adding self loops
                            if graph.x[j, 4] == 0: #is food
                                xy_dist = (x[i]-x[j])[:2]
                                add_edge1(dists[ii][jj], xy_dist, 0)
                            elif dists[ii][jj] <= self.radius: #is below cell radius
                                xy_dist = (x[i]-x[j])[:2]
                                add_edge1(dists[ii][jj], xy_dist, 1)

                time3 = time.perf_counter()

                #zip all dists and indices
                #combine all ii and jj into tuples
                #iterate this single thing - check dist, add edge if constraint

            s_idx = e_idx
            #print('find: ', time2-time1)
            #print('add: ', time3-time2)

        if len(edges) == 0:
            graph.edge_index = torch.tensor([[]], dtype=torch.long, device=self.device)
            graph.edge_attr = torch.tensor([[]], dtype=torch.float, device=self.device)
            return False
        edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=self.device).T
        edge_attributes = np.array(edge_attributes)
        edge_attributes[:, 1:3] = vupdate(edge_attributes[:, 1:3])
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float, device=self.device)
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr
        return True