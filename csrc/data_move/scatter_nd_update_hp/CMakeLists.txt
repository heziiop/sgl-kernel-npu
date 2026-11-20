# Host bridge to the installed CANN ScatterNdUpdate high-performance kernel.
# No OpDef, tiling or device kernel is built by this operator.
sgl_register_operator(
    ID scatter_nd_update_hp
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES scatter_nd_update_hp.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
