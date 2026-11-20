# Included in the HOST target's directory, only when this operator is selected.
# The bridge has a C boundary: its C++ ABI must match nnopbase, not Torch.
find_library(SGL_SCATTER_NNOPBASE nnopbase
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND "${CMAKE_NM}" -D --defined-only "${SGL_SCATTER_NNOPBASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect CANN nnopbase ABI: ${error}")
endif()
if(symbols MATCHES "_ZN2op8internal13GetLogApiInfoB5cxx11Ev")
    set(scatter_cann_abi 1)
elseif(symbols MATCHES "_ZN2op8internal13GetLogApiInfoEv")
    set(scatter_cann_abi 0)
else()
    message(FATAL_ERROR "Cannot determine CANN nnopbase ABI from GetLogApiInfo")
endif()
set(scatter_bridge "${CMAKE_CURRENT_LIST_DIR}/scatter_nd_update_hp.cpp")
set_property(SOURCE "${scatter_bridge}" APPEND PROPERTY COMPILE_OPTIONS
    -U_GLIBCXX_USE_CXX11_ABI "-D_GLIBCXX_USE_CXX11_ABI=${scatter_cann_abi}")
set_property(SOURCE "${scatter_bridge}" APPEND PROPERTY INCLUDE_DIRECTORIES
    "${ASCEND_CANN_PACKAGE_PATH}/include/aclnn")
