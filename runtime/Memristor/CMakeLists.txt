add_library(
    MemristorDialectRuntime
    SHARED
    runtime_c_interface.cpp
    executor.cpp
)

# Loaded with ctypes by cinnamon's torch backend, so it has to be findable
# through the install prefix rather than only in the build tree.
install(TARGETS MemristorDialectRuntime LIBRARY DESTINATION lib)
