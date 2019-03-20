# encoding: utf-8

__author__ = 'Tharun Mathew Paul (tharun@bigpiventures.com)'


class PositionalContextGraphLayer:
    """In this layer the nodes attend on the previous nodes. The number of nodes to attend to
    is specified as an argument.
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

    def get_edges(self, seq_len, edge_id_offset= 0, max_edges=1, **kwargs):
        """Return the edges for this layer."""
        edges = [[] for i in range(self.num_heads)]
        for i in range(0, self.num_heads):
            # Head 1 will attend to prev, Head 2 to next and so on.
            locals = list()
            # For head i, we will attend on node_pos - i if available
            for j in range(0, seq_len):
                if j >= i:
                    locals.append(edge_id_offset + (j * seq_len) + j - i)
            max_layer_eid = max(locals)
            if max_layer_eid > (edge_id_offset + max_edges):
                raise ValueError('Max layer eid {} exceeds {}'.format(max_layer_eid, edge_id_offset + max_edges))
            edges[i] += locals
        return edges
