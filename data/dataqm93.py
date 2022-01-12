import os
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
import random
import functools
import torch
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from torch.functional import F
import dgl
import time


class GraphReader(Dataset):
    def __init__(self, root_dir, rand_seed=123):
        self.root_dir = root_dir
        self.data_dir = "../data/dataset"
        assert os.path.exists(self.data_dir), 'root_dir does not exist'
        with open(self.root_dir) as f:
            self.file_names = f.readlines()
        random.seed(rand_seed)
        random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def init_graph(properties):
        prop = properties.split()

        graph = nx.DiGraph()
        return graph

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):

        with open('../data/atom_pro2.csv', 'r') as apcsv:
            reader = csv.reader(apcsv)
            atom_rows = [row for row in reader]
        filename = self.file_names[idx].strip('\n')
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            try:
                natom = int(f.readline())
            except:
                print(filename)
            properties = f.readline()  # 属性
            graph = GraphReader.init_graph(properties)
            indx = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}
            comp = [0, 0, 0, 0, 0]
            atom_prop = []
            for i in range(natom):
                atom_info = f.readline()
                atom_info = atom_info.split()
                comp[indx[atom_info[0]]-1]+=1
                atom_prop.append(atom_info)
            # print('atom_pro=', atom_pro)
            f.readline()  # 跳过vibrational。。。

            smiles = f.readline()
            smiles = smiles.split()[0]
            since = time.time()
            mol = Chem.MolFromSmiles(smiles)
            assert (mol)
            mol = Chem.AddHs(mol)
            # print(mol)
            mol.RemoveAllConformers()
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)

            fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            features = factory.GetFeaturesForMol(mol)

            #创建结点

            flag = True
            flagM = True

            for i in range(mol.GetNumAtoms()):
                iatom = mol.GetAtomWithIdx(i)
                if iatom.GetSymbol() != atom_prop[i][0]:
                    try:
                        xyz = list(mol.GetConformer().GetAtomPosition(i))
                    except ValueError:
                        flagM = False
                        print(filename, smiles)
                        break
                    else:
                        # print(filename, coord)
                        flag = False
                        break
                else:
                    try:
                        xyz = list(mol.GetConformer().GetAtomPosition(i))
                    except ValueError:
                        flagM = False
                        # print(filename, smiles)
                        break

            if flag:
                for idx in range(mol.GetNumAtoms()):
                    iatom = mol.GetAtomWithIdx(idx)
                    try:
                        co = np.array(atom_prop[idx][1:4]).astype(np.float)
                        graph.add_node(idx, atom_type=iatom.GetSymbol(), coord=co)
                    except ValueError:
                        print(filename)


            for i in range(mol.GetNumAtoms()):
                for j in range(mol.GetNumAtoms()):
                    e_ij = mol.GetBondBetweenAtoms(i, j)
                    distance = np.linalg.norm(graph.nodes[i]['coord'] - graph.nodes[j]['coord'])
                    # distance2 = np.linalg.norm(graph.nodes[i]['coord2'] - graph.nodes[j]['coord2'])
                    if e_ij:
                        graph.add_edge(i, j, bond_type=e_ij.GetBondType(), distance=distance)

                    else:
                        graph.add_edge(i, j, bond_type=None, distance=distance)


        nodes = []
        color = {'H':1,'C':2,'N':3,'O':4,'F':5}
        comp = [i/sum(comp) for i in comp]
        # print(comp)
        colors = []
        for node, ndata in graph.nodes(data=True):
            nfeat = []
            # atom type HCNOF
            # nfeat += [int(node==x) for x in range(11)]
            nfeat += [node]
            # nfeat += [0]
            # nfeat += [float(atom_prop[node][4])]
            nfeat += [int(ndata['atom_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
            colors += [color[ndata['atom_type']]]

            nfeat += [float(i) for i in atom_rows[indx[ndata['atom_type']]][2:]]
            nfeat += comp
            nodes.append(nfeat)

        # edges
        edges = {}
        edge_key = []
        remove_edges = []
        g2_remove_edges = []
        colore = {Chem.rdchem.BondType.SINGLE: 0, Chem.rdchem.BondType.DOUBLE: 1,Chem.rdchem.BondType.TRIPLE:2, Chem.rdchem.BondType.AROMATIC:3}
        colores = []
        label = []
        label_key = []
        g2 = graph.copy(graph)
        # g2 = g2.to_undirected(g2)
        # print("112hang:", graph)
        key_mask = []
        nk_mask = []
        for n1, n2, edata in graph.edges(data=True):
            efeat = []
            if n1 == n2:
                # remove_edges.append((n1, n2))
                g2_remove_edges.append((n1, n2))
                continue;
            if edata['bond_type'] is None:
                g2_remove_edges.append((n1, n2))
                label.append(edata['distance'])
                key_mask.append(0)
                nk_mask.append(1)
                # efeat += [0,0,0,0]
            else:
                label.append(edata['distance'])
                label_key.append(edata['distance'])
                key_mask.append(1)
                nk_mask.append(0)
                efeat += [int(edata['bond_type'] == x) for x in [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC]]
                # print(colore[edata['bond_type']])
                # remove_edges.append((n1, n2))
                colores += [colore[edata['bond_type']]]
            if efeat:
                edges[(n1, n2)] = efeat
                edge_key.append(efeat)
        for e in g2_remove_edges:
            g2.remove_edge(*e)
            # graph.remove_edge(*e)


        for n1, n2, edata in graph.edges(data=True):
            efeat = []
            if n1 == n2:
                remove_edges.append((n1, n2))
                continue
            if edata['bond_type'] is None:
                path = nx.dijkstra_path(g2,n1,n2)
                for i in range(len(path)-1):
                    efeat = list(np.sum([edges[(path[i],path[i+1])], efeat], axis=0))
                # efeat += [0,0,0,0]

            if efeat:
                edges[(n1, n2)] = efeat
                # edge.append(efeat)
        for e in remove_edges:
            graph.remove_edge(*e)

        edge = [value for (key, value) in sorted(edges.items())]

        g = dgl.from_networkx(graph)
        g_key = dgl.from_networkx(g2)

        g.ndata['feature'] = torch.tensor(nodes)
        g.edata['feature'] = torch.tensor(edge)
        g.edata['label'] = torch.tensor(label)
        g.edata['key_mask'] = torch.tensor(key_mask, dtype=torch.bool)
        g.edata['nk_mask'] = torch.tensor(nk_mask, dtype=torch.bool)

        g_key.ndata['feature'] = torch.tensor(nodes)
        g_key.edata['feature'] = torch.tensor(edge_key)
        g_key.edata['label'] = torch.tensor(label_key)
        # g_key.edata['comp'] = torch.tensor(comp)

        x = filename.split('_')
        y = int(x[1].split('.')[0])
        # print(comp)
        return g, g_key, y
        # return g, filename


# if __name__ == "__main__":
#     # data_root = '../data/dataset/'
#     random_seed = 123
#     dataset = GraphReader('../data/rtest.txt', 123)
#     # g, l,k = dataset[0]
#     # print(k)
#     for (i, j, k) in dataset:
#         print(k)
