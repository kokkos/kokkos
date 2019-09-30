################################### FUNCTIONS ##################################
# List of functions
#   kokkos_option

# Validate options are given with correct case and define an internal
# upper-case version for use within 

# 
#
# @FUNCTION: kokkos_deprecated_list
#
# Function that checks if a deprecated list option like Kokkos_ARCH was given.
# This prints an error and prevents configure from completing.
# It attempts to print a helpful message about updating the options for the new CMake.
# Kokkos_${SUFFIX} is the name of the option (like Kokkos_ARCH) being checked.
# Kokkos_${PREFIX}_X is the name of new option to be defined from a list X,Y,Z,...
FUNCTION(kokkos_deprecated_list SUFFIX PREFIX)
  SET(CAMEL_NAME Kokkos_${SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  #I don't love doing it this way but better to be safe
  FOREACH(opt ${KOKKOS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      STRING(REPLACE "," ";" optlist "${${opt}}")
      SET(ERROR_MSG "Given deprecated option list ${opt}. This must now be given as separate -D options, which assuming you spelled options correctly would be:")
      FOREACH(entry ${optlist})
        STRING(TOUPPER ${entry} ENTRY_UC)
        STRING(APPEND ERROR_MSG "\n  -DKokkos_${PREFIX}_${ENTRY_UC}=ON")
      ENDFOREACH()
      STRING(APPEND ERROR_MSG "\nRemove CMakeCache.txt and re-run. For a list of valid options, refer to BUILD.md or even look at CMakeCache.txt (before deleting it).")
      MESSAGE(SEND_ERROR ${ERROR_MSG})
    ENDIF()
  ENDFOREACH()
ENDFUNCTION()

FUNCTION(kokkos_option CAMEL_SUFFIX DEFAULT TYPE DOCSTRING)
  SET(CAMEL_NAME Kokkos_${CAMEL_SUFFIX})
  STRING(TOUPPER ${CAMEL_NAME} UC_NAME)

  # Make sure this appears in the cache with the appropriate DOCSTRING
  SET(${CAMEL_NAME} ${DEFAULT} CACHE ${TYPE} ${DOCSTRING})

  #I don't love doing it this way because it's N^2 in number options, but cest la vie
  FOREACH(opt ${KOKKOS_GIVEN_VARIABLES})
    STRING(TOUPPER ${opt} OPT_UC)
    IF ("${OPT_UC}" STREQUAL "${UC_NAME}")
      IF (NOT "${opt}" STREQUAL "${CAMEL_NAME}")
        MESSAGE(FATAL_ERROR "Matching option found for ${CAMEL_NAME} with the wrong case ${opt}. Please delete your CMakeCache.txt and change option to -D${CAMEL_NAME}=${${opt}}. This is now enforced to avoid hard-to-debug CMake cache inconsistencies.")
      ENDIF()
    ENDIF()
  ENDFOREACH()

  #okay, great, we passed the validation test - use the default
  IF (DEFINED ${CAMEL_NAME})
    SET(${UC_NAME} ${${CAMEL_NAME}} PARENT_SCOPE)
  ELSE()
    SET(${UC_NAME} ${DEFAULT} PARENT_SCOPE)
  ENDIF()

ENDFUNCTION()




