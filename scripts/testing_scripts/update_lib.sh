#!/bin/bash

local machine_input="$1"
local compiler_input="$2"

if [ "$machine_input" = blake ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* || "${ICPCVER}" = 18.0.128 ]]; then
    module swap gcc/4.9.3 gcc/6.4.0
    module list
  fi
fi
