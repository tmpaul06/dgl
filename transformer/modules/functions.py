import torch as th

def src_dot_dst(src_field, dst_field, out_field, head_index=None):
    """
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function.
    """
    def func(edges):
        # If per_head is True then edges are defined per head. We will perform
        # multiplication per head edges[head_index].src[src_field][:,head_index,:] *

        a = edges.src[src_field]
        b = edges.dst[dst_field]
        c = (a * b)
        d = c.sum(-1, keepdim=True)
        if not head_index:
            return {out_field: d}
        else:
            # Per head, for each edge we will have a score. Instead we will now have scores for certain
            # edges for a given edge. Other heads for same edges may or may not update
            return {
                out_field: (edges.src[src_field][:, head_index, :] * edges.dst[dst_field][:, head_index, :]).sum(-1, True)
            }
    return func

def scaled_exp(field, c):
    """
    This function applies $exp(x / c)$ for input $x$, which is required by *Scaled Dot-Product Attention* mentioned in the paper.
    """

    def func(edges):
        return {field: th.exp((edges.data[field] / c).clamp(-10, 10))}
    return func
