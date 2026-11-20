"""CPU-only checks of the real CMake selection and Python reference."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_source(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("scatter_nd_update_hp", {"scatter_nd_update_hp": "HOST"}),
        (
            "scatter_nd_update_hp,layer_norm",
            {"scatter_nd_update_hp": "HOST", "layer_norm": "HOST"},
        ),
        (
            "scatter_nd_update_hp,elu",
            {"scatter_nd_update_hp": "HOST", "elu": "AICORE"},
        ),
    ],
)
def test_native_selection(tmp_path, requested, expected):
    runner = load_source(ROOT / "scripts/validate_native_builds.py")
    selected = runner.discover(tmp_path, "2201", requested)
    assert selected == expected
    scatter_group = next(
        group for group in runner.test_groups({"scatter_nd_update_hp": "HOST"})
    )
    assert scatter_group["impl"] == "cann_ops_nn"
    assert (ROOT / scatter_group["file"]).is_file()


def test_unsupported_architecture(tmp_path):
    runner = load_source(ROOT / "scripts/validate_native_builds.py")
    with pytest.raises(RuntimeError, match="CMake selection failed"):
        runner.discover(tmp_path, "3510", "scatter_nd_update_hp")
    assert (
        "does not support requested ARCH 3510"
        in (tmp_path / "configure.log").read_text()
    )


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_reference_alias_and_untouched_rows(rank, empty):
    torch = pytest.importorskip("torch")
    module = load_source(
        ROOT
        / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_native"
        / "data_move/scatter_nd_update_hp.py"
    )
    shape = (8, 5) if rank == 2 else (8, 1, 5)
    cache = torch.full(shape, 11, dtype=torch.int8)
    slots = [] if empty else [7, 1, 4]
    indices = torch.tensor(slots, dtype=torch.int32).reshape(-1, 1)
    updates = torch.full((len(slots), *shape[1:]), -17, dtype=cache.dtype)
    result = module.scatter_nd_update_hp(cache, indices, updates)
    assert result is cache
    expected = torch.full(shape, 11, dtype=cache.dtype)
    expected[slots] = -17
    assert torch.equal(cache, expected)
