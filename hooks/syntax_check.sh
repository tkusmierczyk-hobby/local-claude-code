#!/usr/bin/env bash
# PostToolUse hook (matcher: Write|Edit).
# Syntax-checks the file Claude just wrote/edited. On failure exits 2 so the
# error is fed straight back to the model — it must fix the file before
# claiming success. Catches the "plausible but broken" edits small local
# models are prone to. Cheap (<100ms), silent when everything is fine.
# Only active under the local-claude launcher (LOCAL_CLAUDE=1), so a cloud
# Claude Code session in the same project is unaffected.

[ "${LOCAL_CLAUDE:-}" = "1" ] || exit 0

file=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)
[ -n "$file" ] && [ -f "$file" ] || exit 0

case "$file" in
  *.py)
    # compile() rather than py_compile, which writes __pycache__ into the project.
    err=$(python3 - "$file" 2>&1 <<'PY'
import sys, traceback
try:
    compile(open(sys.argv[1], "rb").read(), sys.argv[1], "exec")
except (SyntaxError, ValueError) as e:
    sys.exit("".join(traceback.format_exception_only(e)).rstrip())
PY
) || {
      echo "Python syntax check failed for $file:"$'\n'"$err"$'\n'"Fix the syntax error before proceeding." >&2
      exit 2
    } ;;
  *.sh|*.bash)
    err=$(bash -n "$file" 2>&1) || {
      echo "Bash syntax check failed for $file:"$'\n'"$err" >&2
      exit 2
    } ;;
  *.json)
    # Files such as tsconfig.json and .vscode/*.json are JSON with comments and
    # trailing commas, valid where they are used: a file that parses once those
    # are removed passes too. The error reported is the strict parser's.
    err=$(python3 - "$file" 2>&1 <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8").read()
try:
    json.loads(text)
except ValueError as e:
    lenient = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])',
                     lambda m: m.group(1) or "", text, flags=re.S)
    try:
        json.loads(lenient)
    except ValueError:
        sys.exit(str(e))
PY
) || {
      echo "JSON is invalid in $file:"$'\n'"$err" >&2
      exit 2
    } ;;
esac
exit 0
