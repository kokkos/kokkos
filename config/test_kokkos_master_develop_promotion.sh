#!/bin/bash 

. /etc/profile.d/modules.sh

echo "build-dir $1"
echo "backend $2"
echo "module $3"
echo "compiler $4"
echo "cxxflags $5"
echo "kokkosflags $6"
echo "hwloc $7"

NOW=`date "+%Y%m%d%H%M%S"`
BASEDIR="$1-$NOW"

mkdir $BASEDIR
cd $BASEDIR

module load $2

../generate_makefile.sh --with-devices=$2 \
	--compiler=$4 \
	--cxxflags=$5 \
	--with-options=$6

make test
return $?
