import ctypes
import os

import torch
import torch_npu

torch_lib_path = "/usr/local/python3.11.13/lib/python3.11/site-packages"
old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    f"{torch_lib_path}/torch/lib:{torch_lib_path}/torch_npu/lib:" + old_ld_library_path
)

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
shmem_allocator_module_path = os.path.join(current_dir, "output/libshmem_allocator.so")


def switch_to_shmem_allocator():
    new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(shmem_allocator_module_path, "my_malloc", "my_free")
    # Swap the current allocator
    torch_npu.npu.memory.change_current_allocator(new_alloc)
    myallocator = ctypes.CDLL(shmem_allocator_module_path)

    init_fn = ctypes.cast(getattr(myallocator, "my_init"), ctypes.c_void_p).value
    empty_fn = ctypes.cast(getattr(myallocator, "my_empty_cache"), ctypes.c_void_p).value
    get_device_stats_fn = ctypes.cast(getattr(myallocator, "my_get_device_stats"), ctypes.c_void_p).value
    reset_peak_stats_fn = ctypes.cast(getattr(myallocator, "my_reset_peak_stats"), ctypes.c_void_p).value

    new_alloc.allocator().set_init_fn(init_fn)
    new_alloc.allocator().set_reset_fn(empty_fn)
    new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)
    new_alloc.allocator().set_reset_peak_status_fn(reset_peak_stats_fn)


def init_shmem(my_rank, n_ranks, local_mem_size, meta_size, ip_port):
    myallocator = ctypes.CDLL(shmem_allocator_module_path)
    # 设置函数原型
    myallocator.init_shmem.argtypes = [
        ctypes.c_int,      # my_rank
        ctypes.c_int,      # n_ranks
        ctypes.c_uint64,   # local_mem_size
        ctypes.c_uint64,   # meta_size
        ctypes.c_char_p    # ip_port (C字符串)
    ]
    myallocator.init_shmem.restype = None  # void 返回类型

    myallocator.init_shmem(
        ctypes.c_int(my_rank),                    # my_rank
        ctypes.c_int(n_ranks),                    # n_ranks
        ctypes.c_uint64(local_mem_size),  # 1GB local_mem_size
        ctypes.c_uint64(meta_size),                # 4KB meta_size
        ip_port.encode('utf-8')    # ip_port (注意：必须是bytes类型)
    )
