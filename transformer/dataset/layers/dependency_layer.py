# encoding: utf-8

__author__ = 'Tharun Mathew Paul (tharun@bigpiventures.com)'


class DependencyContextGraphLayer:
    """In this layer the nodes attend on the dependency arc heads. Parents also attend on the child
    nodes
    """

    def __init__(self, num_heads=1):
        """

        Parameters
        ---------
        num_heads: int
            The number of heads in the layer. Note that different layers can have different number of heads,
            but we do not enforce that.
        """
        self.num_heads = num_heads

    def dedupe_tuples(self, tups):
        try:
            return list(set([(a, b) if a < b else (b, a) for a, b in tups]))
        except ValueError:
            raise Exception(tups)

    def get_src_dst_deps(self, src_deps, order=1):
        if not isinstance(src_deps, list):
            src_deps = [src_deps]
        # If order is one, then we simply return src_deps
        if order == 1:
            return list(set(src_deps))
        else:
            new_deps = list()
            for src, dst in src_deps:
                # Go up one order. i.e make dst the src, and find its parent
                for src_dup, dst_dup in src_deps:
                    if dst_dup == dst and src != src_dup:
                        new_deps.append((src, src_dup))
                    elif src_dup == src and dst != dst_dup:
                        new_deps.append((dst, dst_dup))
                    elif dst == src_dup and src != dst_dup:
                        new_deps.append((src, dst_dup))
            return list(set(self.get_src_dst_deps(new_deps, order=order - 1)).difference(set(src_deps)))

    def get_edges(self, seq_len, edge_id_offset=0, max_edges=1, **kwargs):
        """Return the edges for this layer."""
        edges = [[] for _ in range(self.num_heads)]
        src_dep = kwargs['src_dep']
        if src_dep:
            for i in range(0, self.num_heads):
                locals = list()
                for src_node_id, dst_node_id in self.dedupe_tuples(self.get_src_dst_deps(src_dep, i + 1)):
                    locals.append(edge_id_offset + src_node_id * seq_len + dst_node_id)
                    locals.append(edge_id_offset + dst_node_id * seq_len + src_node_id)
                max_layer_eid = max(locals)
                if max_layer_eid > (edge_id_offset + max_edges):
                    raise ValueError('Max layer eid {} exceeds {}'.format(max_layer_eid, edge_id_offset + max_edges))
                edges[i] += locals
        return edges
