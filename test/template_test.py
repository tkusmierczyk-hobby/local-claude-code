#!/usr/bin/env python3
"""bin/make-template.sh's prelude on templates other than Qwen's, rendered with jinja2.

Claude Code's conversation arrives as system, user, system. The prelude must
hand the template a single system message at the front, as a plain string:
some templates add a string to it, others read only the first part of a list.
jinja2 stands in for llama.cpp's own template engine here; the real check is
test/e2e_agentic_test.sh against a running server.
"""
import os
import struct
import subprocess
import sys
import tempfile

try:
    import jinja2
except ImportError:
    print("SKIP: jinja2 is not installed")
    sys.exit(0)

MAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "make-template.sh")
TMP = tempfile.mkdtemp()

TEMPLATES = {
    # adds a string to the system content
    "string": "{{ '<sys>' + messages[0]['content'] + '</sys>' }}"
              "{% for m in messages[1:] %}<{{ m.role }}>{{ m.content }}{% endfor %}",
    # Gemma 3 style: a string as it is, otherwise only the first part's text
    "first part": "{% if messages[0]['content'] is string %}{% set s = messages[0]['content'] %}"
                  "{% else %}{% set s = messages[0]['content'][0]['text'] %}{% endif %}"
                  "<sys>{{ s }}</sys>{% for m in messages[1:] %}<{{ m.role }}>{{ m.content }}{% endfor %}",
}
CONVERSATION = [
    {"role": "system", "content": [{"type": "text", "text": "You are a coding agent. "},
                                   {"type": "text", "text": "Follow the rules."}]},
    {"role": "user", "content": "List the files."},
    {"role": "system", "content": " <env>cwd: /tmp/x</env>"},
]
EXPECTED = "<sys>You are a coding agent. Follow the rules. <env>cwd: /tmp/x</env></sys><user>List the files."


def gguf(path, template):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, s("test")), ("tokenizer.chat_template", 8, s(template))]
    with open(path, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
        fh.write(b"".join(s(k) + struct.pack("<I", t) + v for k, t, v in kv))


fails = 0
for name, template in TEMPLATES.items():
    model, out = os.path.join(TMP, "m.gguf"), os.path.join(TMP, "m.jinja")
    gguf(model, template)
    subprocess.run([MAKE, model, out], check=True, capture_output=True)
    try:
        got = jinja2.Environment().from_string(open(out).read()).render(messages=CONVERSATION)
    except Exception as e:
        got = f"{type(e).__name__}: {e}"
    ok = got == EXPECTED
    print(("PASS: " if ok else "FAIL: ") + f"a template that takes the {name} gets every system text, at the front")
    if not ok:
        print(f"        got: {got}")
    fails += not ok

subprocess.run(["rm", "-rf", TMP])
print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
