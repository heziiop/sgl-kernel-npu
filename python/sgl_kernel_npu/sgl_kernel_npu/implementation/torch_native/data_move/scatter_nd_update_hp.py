"""Reference for in-place cache row writes."""


def scatter_nd_update_hp(cache, indices, updates):
    return cache.index_copy_(0, indices.reshape(-1).long(), updates)
