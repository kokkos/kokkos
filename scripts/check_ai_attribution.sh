#!/usr/bin/env bash
# Enforce the Kokkos policy on AI attribution in commits:
#   - AI agents/bots must not be the commit author,
#   - AI agents/bots must not appear in 'Co-authored-by:' or 'Signed-off-by:',
#   - AI assistance is disclosed with 'Assisted-by:' (reported, never an error).
#
# Usage:
#   check_ai_attribution.sh <base>..<head>     check all non-merge commits in range
#   check_ai_attribution.sh --commit-msg FILE  check the commit being created
#                                              (for the pre-commit commit-msg stage)
#
# Exit status: 0 = compliant, 1 = policy violation, 2 = usage error.

set -u

POLICY_URL="https://kokkos.org/kokkos-core-wiki/contributing.html#generative-ai-and-assisted-by-contributions"

# Known generative-AI agents. Deliberately narrow: a bare substring like "bot"
# or "github" would flag humans (e.g. "Talbot",
# "12345+user@users.noreply.github.com").
#
# An identity is an AI agent if its email is a known agent email ...
AI_EMAIL_REGEX='noreply@anthropic\.com|noreply@openai\.com|@agentmail\.to([>[:space:]]|$)|copilot@github\.com|\+copilot@users\.noreply\.github\.com'
# ... or if it has no email at all and its name is a bare agent name, with an
# optional version ("Codex 5") or, for Claude, a model name ("Claude Sonnet 5.5").
# Names are never matched when an email is present, so humans such as
# "Claude Martin <claude.martin@example.org>" or "Devin Smith <...>" pass.
# Known gap: an agent with a non-standard email and a non-"[bot]" name is not caught.
AI_NAME_REGEX='^((claude|copilot|chatgpt|codex|gemini|cursor|devin)( [0-9.]+)*|claude (code|opus|sonnet|haiku|fable)( .*)?)[[:space:]]*$'

# Any other "[bot]" identity (e.g. "claude[bot]", "copilot-swe-agent[bot]") is
# rejected unless it is non-AI automation listed here (matched against the
# identity's name, e.g. "dependabot[bot] <...>").
BOT_ALLOWLIST_REGEX='^(dependabot|pre-commit-ci)\[bot\] <'

# is_ai_identity "Name <email>": succeeds if the identity must not appear as an
# author, co-author or signer.
is_ai_identity() {
  local identity=$1
  grep -Eiq "$AI_EMAIL_REGEX" <<< "$identity" && return 0
  if [[ $identity != *'<'* ]]; then
    grep -Eiq "$AI_NAME_REGEX" <<< "$identity" && return 0
  fi
  grep -Eiq '\[bot\]' <<< "$identity" && ! grep -Eiq "$BOT_ALLOWLIST_REGEX" <<< "$identity"
}

FAILURE_DETECTED=false

# GitHub workflow annotations only make sense inside Actions.
annotate() {
  # $1 = level (error|notice), $2 = title, $3 = message
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo "::$1 title=$2::$3"
  fi
}

report_violation() {
  # $1 = commit label, $2 = title, $3 = offending line, $4... = hints
  local label=$1 title=$2 offending=$3
  shift 3
  annotate error "AI tool policy" "${label:0:12}: ${title}"
  echo "❌ Commit $label: $title"
  echo "    Offending line: $offending"
  local hint
  for hint in "$@"; do
    echo "    $hint"
  done
  FAILURE_DETECTED=true
}

# check_commit LABEL AUTHOR MESSAGE
check_commit() {
  local label=$1 author=$2 message=$3

  # --- POLICY 1: AI AGENTS MUST NOT BE THE COMMIT AUTHOR ---
  if is_ai_identity "$author"; then
    report_violation "$label" "AI agents cannot be the author of a commit." \
      "$author" \
      "Set the author to the human contributor responsible for the change."
  fi

  # --- POLICIES 2 & 3: TRAILERS ---
  local trailer key identity
  while IFS= read -r trailer; do
    [ -n "$trailer" ] || continue

    # Split "Key: Name <email>" so the identity regex sees just the identity.
    key=${trailer%%:*}
    identity=${trailer#*:}
    identity=${identity#"${identity%%[![:space:]]*}"}
    # Lowercase portably; "${key,,}" needs Bash 4 and macOS ships Bash 3.2.
    key=$(printf '%s' "$key" | tr '[:upper:]' '[:lower:]')

    # Informational only: 'Assisted-by:' is the sanctioned disclosure.
    if [ "$key" = "assisted-by" ]; then
      annotate notice "AI assistance disclosed" "${label:0:12}: ${trailer}"
      echo "ℹ️  Commit $label discloses AI assistance: $trailer"
      continue
    fi

    is_ai_identity "$identity" || continue

    case "$key" in
      # --- POLICY 2: AI AGENTS CANNOT BE COAUTHORS ---
      co-authored-by)
        report_violation "$label" "AI agents cannot be listed under 'Co-authored-by:'." \
          "$trailer" \
          "Use 'Assisted-by:' instead."
        ;;
      # --- POLICY 3: AI AGENTS CANNOT SIGN OFF ---
      signed-off-by)
        report_violation "$label" "AI agents cannot sign off on contributions." \
          "$trailer" \
          "'Signed-off-by:' must carry the human contributor's identity."
        ;;
    esac
  done < <(git interpret-trailers --parse --no-divider <<< "$message")
}

check_range() {
  local range=$1 commit

  # Merge commits carry no authored content; skip them.
  local commits
  commits=$(git rev-list --no-merges "$range") || return 2

  if [ -z "$commits" ]; then
    echo "No non-merge commits to check."
    return 0
  fi

  for commit in $commits; do
    check_commit "$commit" \
      "$(git log -1 --format='%an <%ae>' "$commit")" \
      "$(git log -1 --format='%B' "$commit")"
  done
}

check_commit_msg_file() {
  local file=$1 ident message

  if [ ! -r "$file" ]; then
    echo "error: cannot read commit message file '$file'" >&2
    return 2
  fi

  # "Name <email> <timestamp> <tz>" -> "Name <email>"
  ident=$(git var GIT_AUTHOR_IDENT) || return 2
  ident=${ident% * *}

  # The hook sees the raw message, including editor comment lines.
  message=$(git stripspace --strip-comments < "$file")

  check_commit "commit-msg" "$ident" "$message"
}

usage() {
  echo "usage: $0 <base>..<head> | --commit-msg FILE" >&2
}

case "${1:-}" in
  --commit-msg)
    [ $# -eq 2 ] || { usage; exit 2; }
    check_commit_msg_file "$2" || exit 2
    ;;
  ""|-*)
    usage
    exit 2
    ;;
  *)
    [ $# -eq 1 ] || { usage; exit 2; }
    check_range "$1" || exit 2
    ;;
esac

if [ "$FAILURE_DETECTED" = true ]; then
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo "::error::Commit validation failed. Human attribution rules were violated."
  fi
  echo "See policy at $POLICY_URL"
  exit 1
fi
echo "✅ All commits follow the human authorship and AI attribution guidelines."
