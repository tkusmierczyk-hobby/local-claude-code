#!/usr/bin/env bash
# make-template.sh — write a chat template that tolerates Claude Code's payload.
#
# Claude Code sends its environment/context block as a *trailing* system message
# (roles come through as system, user, system). Qwen's stock template raises
# "System message must be at the beginning." and llama-server returns HTTP 500,
# so every request fails once Claude Code is the client rather than curl.
#
# The fix is a prelude prepended to the model's own template: it merges every
# system message into a single one at the front and leaves the rest untouched.
# Nothing else about the template changes, so tool calls and thinking still work.
#
# Usage: make-template.sh <model.gguf> <out.jinja>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?usage: make-template.sh <model.gguf> <out.jinja>}"
OUT="${2:?usage: make-template.sh <model.gguf> <out.jinja>}"

mkdir -p "$(dirname "$OUT")"
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT

cat > "$TMP" <<'PRELUDE'
{#- local-claude: merge all system messages into one at the front. Claude Code
    sends a trailing system message and this model's template rejects that.
    The merged content is a plain string: some templates add a string to it,
    others read only the first part of a list. Texts are joined with nothing in
    between, as Qwen's own template joins the parts of a list, so its prompt
    is unchanged. A namespace is required: plain `set` inside a for loop does
    not survive it. The last tag ends in `-%}` so that no newline from here
    reaches the prompt, ahead of a template that starts by printing bos_token. -#}
{%- set lc = namespace(sys="", has_sys=false, rest=[]) %}
{%- for lc_m in messages %}
    {%- if lc_m.role == "system" %}
        {%- set lc.has_sys = true %}
        {%- if lc_m.content is string %}
            {%- set lc.sys = lc.sys ~ lc_m.content %}
        {%- elif lc_m.content is not none %}
            {%- for lc_p in lc_m.content %}
                {%- if 'text' in lc_p %}
                    {%- set lc.sys = lc.sys ~ lc_p.text %}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {%- else %}
        {%- set lc.rest = lc.rest + [lc_m] %}
    {%- endif %}
{%- endfor %}
{%- if lc.has_sys %}
    {%- set messages = [{"role": "system", "content": lc.sys}] + lc.rest %}
{%- else %}
    {%- set messages = lc.rest %}
{%- endif -%}
PRELUDE

python3 "$HERE/patch-gguf.py" "$MODEL" --template >> "$TMP"
mv -f "$TMP" "$OUT"
trap - EXIT
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
