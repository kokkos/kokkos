#!/usr/bin/env bash

set -exo pipefail

if test $# -lt 1
then
    echo "usage: ./$0 <cmake-version>"
    exit 1
fi

cmake_version=$1
cmake_tar_name=cmake-${cmake_version}-Linux-x86_64.tar.gz

echo "${cmake_version}"
echo "${cmake_tar_name}"

wget http://github.com/Kitware/CMake/releases/download/v${cmake_version}/${cmake_tar_name}

tar xzf ${cmake_tar_name} --one-top-level=cmake --strip-components 1
