################################### FUNCTIONS ##################################
# List of functions
#   set_kokkos_cxx_compiler
#   set_kokkos_cxx_standard
#   kokkos_option

FUNCTION(kokkos_option CAMEL_SUFFIX DEFAULT TYPE DOCSTRING)
  SET(CAMEL_NAME Kokkos_${CAMEL_SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)
  SET(CACHE_NAME KOKKOS_CACHED_${UC_NAME})
  IF (NOT DEFINED ${CACHE_NAME} AND DEFINED ${CAMEL_NAME})
    #THIS IS our first time through the cmake
    #WE WERE given the camel case name instead of the UC name we wanted
    #MAKE DARn sure we don't have both an UC and Camel version that differ
    IF (DEFINED ${UC_NAME} AND NOT ${CAMEL_NAME} STREQUAL ${UC_NAME})
      MESSAGE(FATAL_ERROR "Given both ${CAMEL_NAME} and ${UC_NAME} with different values: ${${CAMEL_NAME}} != ${${UC_NAME}}")
    ENDIF()
    #GREAT, No conflicts - use the camel case name as the default for the UC
    SET(${UC_NAME} ${${CAMEL_NAME}} CACHE ${TYPE} ${DOCSTRING})
  ELSEIF(DEFINED ${CAMEL_NAME})
    #THIS IS at least our second configure and we have an existing cache
    #CMAKE Makes this impossible to distinguish something already in cache
    #AND SOMthing given explicitly on the command line
    #AT THIS point, we have no choice but to accept the Camel value and print a warning
    IF (NOT ${CAMEL_NAME} STREQUAL ${UC_NAME})
      MESSAGE(WARNING "Overriding ${UC_NAME}=${${UC_NAME}} with ${CAMEL_NAME}=${${CAMEL_NAME}}")
    ENDIF()
    #I HAVE to accept the Camel case value - really no choice here - force it
    SET(${UC_NAME} ${${CAMEL_NAME}} CACHE ${TYPE} ${DOCSTRING} FORCE)
  ELSE() #GReat, no camel case names - nice and simple
    SET(${UC_NAME} ${DEFAULT} CACHE ${TYPE} ${DOCSTRING})
  ENDIF()
  #STORE A Value in the cache to identify whether this is the 1st configure
  SET(${CACHE_NAME} ${${UC_NAME}} CACHE ${TYPE} ${DOCSTRING} FORCE)

  IF (${UC_NAME}) #cmake if statements follow really annoying string resolution rules
    MESSAGE(STATUS "${UC_NAME}=${${UC_NAME}}") 
  ENDIF()
ENDFUNCTION(kokkos_option)




