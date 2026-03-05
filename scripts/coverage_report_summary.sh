#!/bin/sh
# Print header and summary (last line) from llvm-cov coverage_report.txt as markdown table
# Usage: coverage_report_summary.sh [coverage_report.txt]
# Default: coverage_report.txt in current directory

REPORT="${1:-coverage_report.txt}"
if [ ! -f "$REPORT" ]; then
  echo "Error: $REPORT not found" >&2
  exit 1
fi

_header=$(head -1 "$REPORT")
_data=$(tail -1 "$REPORT")

# Convert space-separated columns to markdown table
# Splits on 2+ spaces to handle llvm-cov column alignment
_to_md_row() {
  printf '%s' "$1" | awk '
    BEGIN { FS = "  +" }
    {
      $1 = $1
      if (NF > 0) {
        printf "|"
        for (i = 1; i <= NF; i++) printf " %s |", $i
        printf "\n"
      }
    }'
}

_sep_row() {
  printf '%s' "$1" | awk '
    BEGIN { FS = "  +" }
    {
      $1 = $1
      if (NF > 0) {
        printf "|"
        for (i = 1; i <= NF; i++) printf " --- |"
        printf "\n"
      }
    }'
}

_to_md_row "$_header"
_sep_row "$_header"
_to_md_row "$_data"
