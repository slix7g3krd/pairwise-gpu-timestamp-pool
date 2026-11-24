import pytest
import graphics.gpu_timestamp_pool as mod

CASES = [
    ("rotate_pool", (1,)),
    ("readback_bind", (1,)),
    ("same_queue", (1,)),
    ("scope_guard", (1,)),
    ("release_old", (1,)),
    ("grow_preserve", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("query_begin", (1,)),
    ("metrics", (1,)),
    ("query_end", (1,)),
    ("resolve", (1,)),
    ("stats", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
