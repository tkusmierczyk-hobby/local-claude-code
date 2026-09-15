#!/usr/bin/env bash
# PostToolUse hook (matcher: Write|Edit).
# Syntax-checks the file Claude just wrote/edited. On failure exits 2 so the
# error is fed straight back to the model — it must fix the file before
# claiming success. Catches the "plausible but broken" edits small local
# models are prone to. Cheap (<100ms), silent when everything is fine.

file=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)
[ -n "$file" ] && [ -f "$file" ] || exit 0

case "$file" in
  *.py)
    err=$(python3 -m py_compile "$file" 2>&1) || {
      echo "Python syntax check failed for $file:"$'\n'"$err"$'\n'"Fix the syntax error before proceeding." >&2
      exit 2
    } ;;
  *.sh|*.bash)
    err=$(bash -n "$file" 2>&1) || {
      echo "Bash syntax check failed for $file:"$'\n'"$err" >&2
      exit 2
    } ;;
  *.json)
    err=$(python3 -m json.tool "$file" 2>&1 >/dev/null) || {
      echo "JSON is invalid in $file:"$'\n'"$err" >&2
      exit 2
    } ;;
esac
exit 0
