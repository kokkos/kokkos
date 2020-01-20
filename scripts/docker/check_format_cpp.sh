#!/usr/bin/env bash

./scripts/apply-clang-format || exit $?
git diff --exit-code
