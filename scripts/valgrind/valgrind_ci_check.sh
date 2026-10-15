#!/usr/bin/bash

TMPFILE=$(mktemp)

set -o pipefail

ctest -D ExperimentalMemCheck \
	--overwrite "MemoryCheckCommand=/usr/bin/valgrind" \
	--overwrite "MemoryCheckCommandOptions=--leak-check=full --show-leak-kinds=all --quiet --suppressions=$PWD/../scripts/valgrind/kokkos.supp" \
	--output-on-failure | tee "$TMPFILE"

EXIT_CODE=$?

set +o pipefail

LEAKED_TESTS=$(sed -nE "s/.* MemCheck: #([0-9]+).* Defects: [0-9]+/Testing\/Temporary\/MemoryChecker.\1.log/p" "$TMPFILE")

if [ -z "$LEAKED_TESTS" ]; then
	exit $EXIT_CODE
fi

echo ""
echo "--------------------------------------"
echo "------------ Stack traces ------------"
echo "--------------------------------------"

for FILE in $LEAKED_TESTS;
do
	echo ""
	echo "------------ $FILE ------------"
	echo ""
	cat "$FILE"
done

exit 1
