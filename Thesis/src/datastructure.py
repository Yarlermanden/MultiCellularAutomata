import grispy as gsp
import numpy as np
import torch
import time
from torch import Tensor
from sklearn.neighbors import KDTree

from graphUtils import *

class FixedRadiusNearestNeighbors2(object):
    def __init__(self, nodes, dense):
        nodes = nodes.detach().cpu().numpy()
        if dense: self.tree = KDTree(nodes, leaf_size=40)
        else: self.tree = KDTree(nodes, leaf_size=20)

    def get_neighbors(self, node, radius):
        return self.tree.query_radius(node.detach().cpu().numpy(), radius, return_distance=True)

class DataStructure(object):
    def __init__(self, settings):
        self.settings = settings
        self.device = settings.device
        self.wrap_around = settings.wrap_around
        self.batch_size = settings.batch_size
        self.scale = settings.scale

    def update_dist1(self, dist):
        if dist > self.scale:
            return -2*self.scale + dist
        elif dist < -self.scale:
            return 2*self.scale + dist
        return dist

    def update_norm_dist(self, norm):
        if norm > 1.4142135624:
            return -2.828427 + norm
        elif norm < -1.4142135624:
            return 2.828427 + norm
        return norm

    def add_edges(self, graph):
        vupdate = np.vectorize(self.update_dist1)
        edges = []

        x = graph.x[:, :2].detach().cpu().numpy()
        s_idx = 0
        for batch_idx in range(self.batch_size): #TODO could we simply just vectorize this entire thing?
            e_idx = s_idx + graph.subsize[batch_idx].detach().cpu().numpy()
            nodes = graph.x[s_idx:e_idx, :2]
            if len(nodes) == 0: continue
            cell_indices = torch.nonzero(cell_mask(graph.x[s_idx:e_idx])).flatten()
            cells = nodes[cell_indices]
            food_indices = torch.nonzero(food_mask(graph.x[s_idx:e_idx])).flatten() #it's on purpose these are not added with s_idx - we want only relative indices
            food = nodes[food_indices]
            wall_indices = torch.nonzero(wall_mask(graph.x[s_idx:e_idx])).flatten()
            walls = nodes[wall_indices]

            if len(cells) != 0:
                cell_indices += s_idx
                food_indices += s_idx
                wall_indices += s_idx
                cell_indices = cell_indices.detach().cpu().numpy()
                food_indices = food_indices.detach().cpu().numpy()
                wall_indices = wall_indices.detach().cpu().numpy()

                frnn_food = FixedRadiusNearestNeighbors2(food, False)
                indices_food, dists_food = frnn_food.get_neighbors(cells, self.settings.radius_food)
                indices_food = [food_indices[x] for x in indices_food]
                edges_food = [[j, i, dists_food[ii][jj], *(x[i]-x[j]), EdgeType.FoodToCell]
                            for ii, i in enumerate(cell_indices) for jj, j in enumerate(indices_food[ii])]

                radius_cell = torch.where(graph.x[cell_indices, 3] == 3, self.settings.radius_long, self.settings.radius_cell)
                frnn_cell = FixedRadiusNearestNeighbors2(cells, True)
                indices_cells, dists_cells = frnn_cell.get_neighbors(cells, radius_cell.detach().cpu().numpy())                       
                indices_cells = [cell_indices[x] for x in indices_cells]
                edges_cells = [[j, i, dists_cells[ii][jj], *(x[i]-x[j]), EdgeType.CellToCell]
                       for ii, i in enumerate(cell_indices) for jj, j in enumerate(indices_cells[ii])
                       if i!=j]
                
                frnn_wall = FixedRadiusNearestNeighbors2(walls, False)
                indices_walls, dists_walls = frnn_wall.get_neighbors(cells, self.settings.radius_wall)
                indices_walls = [wall_indices[x] for x in indices_walls]
                edges_walls = [[j, i, dists_walls[ii][jj], *(x[i]-x[j]), EdgeType.WallToCell]
                        for ii, i in enumerate(cell_indices) for jj, j in enumerate(indices_walls[ii])]

                if len(edges_food) > 0:
                    edges.extend(edges_food)
                if len(edges_cells) > 0:
                    edges.extend(edges_cells)
                if len(edges_walls) > 0:
                    edges.extend(edges_walls)
            s_idx = e_idx
         
        if len(edges) == 0:
            graph.edge_index = torch.tensor([[]], dtype=torch.long, device=self.device)
            graph.edge_attr = torch.tensor([[]], dtype=torch.float, device=self.device)
            return False
        edges = np.array(edges)
        graph.edge_index = torch.tensor(edges[:, :2], dtype=torch.long, device=self.device).T
        edge_attributes = edges[:, 2:]
        edge_attributes[:, 1:3] = vupdate(edge_attributes[:, 1:3]) #restrict to match wraparound
        graph.edge_attr = torch.tensor(edge_attributes, dtype=torch.float, device=self.device)
        return True

    def add_edges_with_global_node(self, graph):
        '''Add edges according to a organism containing a global node for long range communication'''
        vupdate = np.vectorize(self.update_dist1)
        vcompute_dist = np.vectorize(self.update_norm_dist)
        any = self.add_edges(graph)
        if not any: return False
        edges = []
        attributes = []

        s_idx = 0
        for batch_idx in range(self.batch_size):
            e_idx = s_idx + graph.subsize[batch_idx].detach().cpu().numpy()

            nodes = graph.x[s_idx:e_idx]
            cell_idx = torch.nonzero(cell_mask(nodes)).flatten()
            cells = nodes[cell_idx, :2]
            global_node_idx = torch.nonzero(nodes[:, 4] == 2).squeeze()
            global_node = nodes[global_node_idx, :2].squeeze()

            cell_idx+=s_idx
            global_node_idx+=s_idx

            tup = [([cell_idx[i], global_node_idx], 
                    [0, 0, 0, EdgeType.GlobalAndCell],
                    [global_node_idx, cell_idx[i]], 
                    [0, 0, 0, EdgeType.GlobalAndCell])
                   for i in range(len(cells))]
            l = [list(t) for t in zip(*tup)]
            if len(l) > 0:
                edges.extend(l[0])
                edges.extend(l[2])
                attributes.extend(l[1])
                attributes.extend(l[3])
            s_idx = e_idx

        if len(edges) == 0:
            return False
        edge_attributes = np.array(attributes)
        edge_attributes[:, 0] = np.linalg.norm(edge_attributes[:, 1:3], axis=1)
        edge_attributes[:, 0] = vcompute_dist(edge_attributes[:, 0])
        edge_attributes[:, 1:3] = vupdate(edge_attributes[:, 1:3]) #restrict to match wraparound

        graph.edge_index = torch.concat((torch.tensor(np.array(edges), dtype=torch.long, device=self.device).T, graph.edge_index), dim=1)
        graph.edge_attr = torch.concat((torch.tensor(edge_attributes, dtype=torch.float, device=self.device), graph.edge_attr), dim=0)
        return True
        
    def add_edges_rigid(self, graph):
        '''Add edges according to a rigid organism, where all cells stay connected'''
        ...
        #don't need to readd edges for cell to cell as these are rigid
        #Only need to add edges for food