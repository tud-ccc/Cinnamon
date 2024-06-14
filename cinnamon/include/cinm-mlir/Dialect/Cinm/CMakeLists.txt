################################################################################
# CinmIncGen
#
# MLIR Cinm dialect generated includes.
################################################################################

add_custom_target(CinmIncGen)

# Attributes, Dialect, Operations and Types.
add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)
