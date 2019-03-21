import dgl
import torch as th
import numpy as np
import itertools
import time
from collections import *
import itertools
from .layers import PositionalContextGraphLayer, DependencyContextGraphLayer

Graph = namedtuple('Graph',
                   ['g', 'src', 'tgt', 'tgt_y', 'nids', 'eids', 'nid_arr', 'n_nodes', 'n_edges', 'n_tokens', 'layer_eids'])

# We need to create new graph pools for relative position attention (ngram style)


class GraphPool:
    "Create a graph pool in advance to accelerate graph building phase in Transformer."
    def __init__(self, n=50, m=50):
        '''
        args:
            n: maximum length of input sequence.
            m: maximum length of output sequence.
        '''
        print('start creating graph pool...')
        tic = time.time()
        self.n, self.m = n, m
        g_pool = [[dgl.DGLGraph() for _ in range(m)] for _ in range(n)]
        num_edges = {
            'ee': np.zeros((n, n)).astype(int),
            'ed': np.zeros((n, m)).astype(int),
            'dd': np.zeros((m, m)).astype(int)
        }
        for i, j in itertools.product(range(n), range(m)):
            src_length = i + 1
            tgt_length = j + 1

            g_pool[i][j].add_nodes(src_length + tgt_length)
            enc_nodes = th.arange(src_length, dtype=th.long)
            dec_nodes = th.arange(tgt_length, dtype=th.long) + src_length

            # enc -> enc
            us = enc_nodes.unsqueeze(-1).repeat(1, src_length).view(-1)
            vs = enc_nodes.repeat(src_length)

            g_pool[i][j].add_edges(us, vs)
            num_edges['ee'][i][j] = len(us)
            # enc -> dec
            us = enc_nodes.unsqueeze(-1).repeat(1, tgt_length).view(-1)
            vs = dec_nodes.repeat(src_length)
            g_pool[i][j].add_edges(us, vs)
            num_edges['ed'][i][j] = len(us)
            # dec -> dec
            indices = th.triu(th.ones(tgt_length, tgt_length)) == 1
            us = dec_nodes.unsqueeze(-1).repeat(1, tgt_length)[indices]
            vs = dec_nodes.unsqueeze(0).repeat(tgt_length, 1)[indices]
            g_pool[i][j].add_edges(us, vs)
            num_edges['dd'][i][j] = len(us)

        print('successfully created graph pool, time: {0:0.3f}s'.format(time.time() - tic))
        self.g_pool = g_pool
        self.num_edges = num_edges

    def beam(self, src_buf, start_sym, max_len, k, n_layers, n_heads, device='cpu', src_deps=None, vocab=None):
        '''
        Return a batched graph for beam search during inference of Transformer.
        args:
            src_buf: a list of input sequence
            start_sym: the index of start-of-sequence symbol
            max_len: maximum length for decoding
            k: beam size
            n_layers: the number of layers
            n_heads: the number of heads
            device: 'cpu' or 'cuda:*' 
        '''
        if src_deps is None:
            src_deps = list()
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [max_len] * len(src_buf)
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            for _ in range(k):
                g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = dgl.batch(g_list)
        src, tgt = [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        layer_eids = [
            [
                [] for _ in range(0, n_heads)
            ] for _ in range(0, n_layers)
        ]
        e2e_eids, e2d_eids, d2d_eids = [], [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, src_dep, n, n_ee, n_ed, n_dd in zip(src_buf, src_deps, src_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            for _ in range(k):
                src.append(th.tensor(src_sample, dtype=th.long, device=device))
                src_pos.append(th.arange(n, dtype=th.long, device=device))
                enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
                n_nodes += n
                e2e_eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))

                # Copy the ids of edges that correspond to a given node and its previous N nodes
                # We are using arange here. This will not work. Instead we need to select edges that
                # correspond to previous positions. This information is present in graph pool
                # For each edge, we need to figure out source_node_id and target_node_id.

                custom_edges = self.get_edges_per_layer(n_heads, src_dep, n_edges, n_ee, n, device)
                for ii, jj in itertools.product(range(n_layers), range(n_heads)):
                    layer_eids[ii][jj] += (custom_edges[ii][jj])

                n_edges += n_ee
                tgt_seq = th.zeros(max_len, dtype=th.long, device=device)
                tgt_seq[0] = start_sym
                tgt.append(tgt_seq)
                tgt_pos.append(th.arange(max_len, dtype=th.long, device=device))

                dec_ids.append(th.arange(n_nodes, n_nodes + max_len, dtype=th.long, device=device))
                n_nodes += max_len

                e2d_eids.append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
                n_edges += n_ed
                d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
                n_edges += n_dd

        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=None,
                     nids = {'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids = {'ee': th.cat(e2e_eids), 'ed': th.cat(e2d_eids), 'dd': th.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     layer_eids=layer_eids,
                     n_tokens=n_tokens)

    def get_edges_per_layer(self, num_layers, num_heads, src_dep, n_edges, n_ee, n, device):
        """Return edges for each head in a given layer"""
        edges_per_layer = list()

        # layer 0: Positional
        edges_per_layer.append(PositionalContextGraphLayer(num_heads).get_edges(
            n,
            edge_id_offset=n_edges,
            max_edges=n_ee
        ))

        # layer 1: Dependency
        edges_per_layer.append(DependencyContextGraphLayer(num_heads).get_edges(
            n,
            edge_id_offset=n_edges,
            max_edges=n_ee,
            src_dep=src_dep
        ))

        if num_layers > 2:
            # Add None so that regular edges are used
            for i in range(2, num_layers):
                edges_per_layer.append(
                    [list(range(n_edges, n_edges + n_ee)) for _ in range(num_heads)]
                )

        return edges_per_layer

    def __call__(self, src_buf, tgt_buf, n_layers, n_heads, device='cpu', src_deps=None, vocab=None):
        '''
        Return a batched graph for the training phase of Transformer.
        args:
            src_buf: a set of input sequence arrays.
            tgt_buf: a set of output sequence arrays.
            n_layers: the number of layers
            n_heads: the number of heads
            device: 'cpu' or 'cuda:*'
            src_deps: list, optional
                Dependency parses of the source in the form of src_node_id -> dst_node_id.
                where src is the child and dst is the parent. i.e a child node attends on its
                syntactic parent in a dependency parse
        '''

        if src_deps is None:
            src_deps = list()
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [len(_) - 1 for _ in tgt_buf]

        num_edges = {'ee': [], 'ed': [], 'dd': []}

        # We are running over source and target pairs here
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = dgl.batch(g_list)
        src, tgt, tgt_y = [], [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        e2e_eids, d2d_eids, e2d_eids = [], [], []
        # For each layer, we have num_heads number of eids
        layer_eids = [
            [
                [] for _ in range(0, n_heads)
            ] for _ in range(0, n_layers)
        ]
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, tgt_sample, src_dep, n, m, n_ee, n_ed, n_dd in zip(src_buf, tgt_buf, src_deps, src_lens, tgt_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            src.append(th.tensor(src_sample, dtype=th.long, device=device))
            tgt.append(th.tensor(tgt_sample[:-1], dtype=th.long, device=device))
            tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
            src_pos.append(th.arange(n, dtype=th.long, device=device))
            tgt_pos.append(th.arange(m, dtype=th.long, device=device))
            enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
            n_nodes += n
            dec_ids.append(th.arange(n_nodes, n_nodes + m, dtype=th.long, device=device))
            n_nodes += m

            e2e_eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))

            # Copy the ids of edges that correspond to a given node and its previous N nodes
            # We are using arange here. This will not work. Instead we need to select edges that
            # correspond to previous positions. This information is present in graph pool
            # For each edge, we need to figure out source_node_id and target_node_id.
            custom_edges = self.get_edges_per_layer(n_layers, n_heads, src_dep, n_edges, n_ee, n, device)
            for ii, jj in itertools.product(range(n_layers), range(n_heads)):
                layer_eids[ii][jj] += (custom_edges[ii][jj])

            n_edges += n_ee
            e2d_eids.append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
            n_edges += n_ed
            d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
            n_edges += n_dd
            n_tokens += m


        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=th.cat(tgt_y),
                     nids = {'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids = {'ee': th.cat(e2e_eids), 'ed': th.cat(e2d_eids), 'dd': th.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     # Concatenate
                     layer_eids=layer_eids,
                     n_edges=n_edges,
                     n_tokens=n_tokens)
