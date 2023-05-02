# @HEADER
# ************************************************************************
#
#            TriBITS: Tribal Build, Integrate, and Test System
#                    Copyright 2013 Sandia Corporation
#
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ************************************************************************
# @HEADER


# @MACRO: tribits_pkg_export_cache_var()
#
# Macro that registers a package-level cache var to be exported in the
# ``<Package>Config.cmake`` file
#
# Usage::
#
#   tribits_pkg_export_cache_var(<cacheVarName>)
#
# where ``<cacheVarName>`` must be the name of a cache variable (or an error
# will occur).
#
# NOTE: This will also export this variable to the
# ``<Package><Spkg>Config.cmake`` file for every enabled subpackage (if this
# is called from a ``CMakeLists.txt`` file of a top-level package that has
# subpackages).  That way, any top-level package cache vars are provided by
# any of the subpackages' ``<Package><Spkg>Config.cmake`` files.
#
macro(tribits_pkg_export_cache_var  cacheVarName)
  if (DEFINED ${PACKAGE_NAME}_PKG_VARS_TO_EXPORT)
    # Assert this is a cache var
    get_property(cacheVarIsCacheVar  CACHE ${cacheVarName} PROPERTY  VALUE  SET)
    if (NOT cacheVarIsCacheVar)
      message(SEND_ERROR
        "ERROR: The variable ${cacheVarName} is NOT a cache var and cannot"
        " be exported!")
    endif()
    # Add to the list of package cache vars to export
    set(${PACKAGE_NAME}_PKG_VARS_TO_EXPORT
      "${${PACKAGE_NAME}_PKG_VARS_TO_EXPORT};${cacheVarName}" CACHE INTERNAL "")
  endif()
endmacro()


# Function that sets up data-structures for package-level cache var to be
# exported
#
function(tribits_pkg_init_exported_vars  PACKAGE_NAME_IN)
  set(${PACKAGE_NAME_IN}_PKG_VARS_TO_EXPORT "" CACHE INTERNAL "")
endfunction()


# Function that injects set() statements for a package's exported cache vars into
# a string.
#
# This is used to create set() statements to be injected into a package's
# ``<Package>Config.cmake`` file.
#
function(tribits_pkg_append_set_commands_for_exported_vars  packageName
    configFileStrInOut
  )
  set(configFileStr "${${configFileStrInOut}}")
  foreach(exportedCacheVar IN LISTS ${packageName}_PKG_VARS_TO_EXPORT)
    if (NOT "${exportedCacheVar}" STREQUAL "")
      string(APPEND configFileStr
        "set(${exportedCacheVar} \"${${exportedCacheVar}}\")\n")
    endif()
  endforeach()
  set(${configFileStrInOut} "${configFileStr}" PARENT_SCOPE)
endfunction()
