# encoding: utf-8
import pytest

from ..graph import GraphPool

__author__ = 'Tharun Mathew Paul (tharun@bigpiventures.com)'


@pytest.mark.unit
def test_graph_pool_src_deps():
    pool = GraphPool(3, 3)

    result = pool([[1, 2, 3], [3, 4, 5]], [[11, 12, 13], [18, 92, 15]], src_deps=[[(0, 2), (0, 1)], [(0, 1)]])

    assert result.layer_eids['dep'][0] == [[1, 3, 2, 6], [5, 7], []]
