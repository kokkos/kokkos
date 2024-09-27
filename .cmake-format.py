# -----------------------------
# Options affecting formatting.
# -----------------------------
with section("format"):

  # How wide to allow formatted cmake files
  line_width = 120

  # If an argument group contains more than this many sub-groups (parg or kwarg
  # groups) then force it to a vertical layout.
  max_subgroups_hwrap = 3

  # If a statement is wrapped to more than one line, than dangle the closing
  # parenthesis on its own line.
  dangle_parens = True

  # If the trailing parenthesis must be 'dangled' on its on line, then align it
  # to this reference: `prefix`: the start of the statement,  `prefix-indent`:
  # the start of the statement, plus one indentation  level, `child`: align to
  # the column of the arguments
  dangle_align = 'prefix'

# ------------------------------------------------
# Options affecting comment reflow and formatting.
# ------------------------------------------------
with section("markup"):
  # enable comment markup parsing and reflow
  enable_markup = False
