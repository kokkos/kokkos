#!/usr/bin/env bash

clang_format_executable=${CLANG_FORMAT_EXE:-clang-format}

this_program=$(basename "$0")
usage="Usage:
  $this_program [options] -- check format of the C++ source files

Options:
    -h --help         Print help and exit
    -q --quiet        Quiet mode (do not print the diff)
    -p --apply-patch  Apply diff patch to the source files"

verbose=1
apply_patch=0

#echo "Arguments: $# $@"

while [ $# -gt 0 ]
do
  case $1 in
  -p|--apply-patch)
      apply_patch=1
      ;;
  -q|--quiet)
      verbose=0
      ;;
  -h|--help)
      echo "$usage"
      exit 0
      ;;
  *)
      echo "$this_program: Unknown argument '$1'. See '$this_program --help'."
      exit -1
      ;;
  esac

  shift
done

# stop right here if clang-format does not exist in $PATH
command -v $clang_format_executable >/dev/null 2>&1 || { echo >&2 "clang-format executable '$clang_format_executable' not found.  Aborting."; exit 1; }

# shamelessy redirecting everything to /dev/null in quiet mode
if [ $verbose -eq 0 ]; then
    exec &>/dev/null
fi

cpp_source_files=$(git ls-files | grep -E "\.hpp$|\.cpp$|\.h$|\.c$" | grep -v -f .clang-format-ignore)

unformatted_files=()
for file in $cpp_source_files; do
    diff -u \
        <(cat $file) \
        --label a/$file \
        <($clang_format_executable $file) \
        --label b/$file >&1
    if [ $? -eq 1 ]; then
        unformatted_files+=($file)
    fi
done

n_unformatted_files=${#unformatted_files[@]}
if [ $n_unformatted_files -ne 0 ]; then
    echo "${#unformatted_files[@]} file(s) not formatted properly:"
    for file in ${unformatted_files[@]}; do
        echo "    $file"
        if [ $apply_patch -eq 1 ]; then
            $clang_format_executable -i $file
        fi
    done
else
    echo "OK"
fi
exit $n_unformatted_files
