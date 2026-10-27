#!/usr/bin/env python3
"""bin/patch-gguf.py on a tiny GGUF: one patched key, tensor data intact, input never overwritten."""
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "patch-gguf.py")
TMP = tempfile.mkdtemp()


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def make_gguf(path):
    """An ollama-style qwen35moe header (3 rope sections) and one 4-float tensor."""
    kv = [("general.architecture", 8, s("qwen35moe")),
          ("qwen35moe.rope.dimension_sections", 9, struct.pack("<IQ", 5, 3) + struct.pack("<3i", 11, 11, 10)),
          ("tokenizer.chat_template", 8, s("{{ messages }}"))]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv))
    head += b"".join(s(k) + struct.pack("<I", t) + v for k, t, v in kv)
    head += s("t") + struct.pack("<IQIQ", 1, 4, 0, 0)  # name, n_dims, dims, type F32, offset
    head += b"\0" * (-len(head) % 32)
    with open(path, "wb") as fh:
        fh.write(head + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


fails = 0


def check(name, cond):
    global fails
    print(("PASS: " if cond else "FAIL: ") + name)
    fails += not cond


src, out = os.path.join(TMP, "in.gguf"), os.path.join(TMP, "out.gguf")
make_gguf(src)
before = digest(src)

r = run(src, out, "--fix-ollama")
shown = run(out, "--show").stdout
with open(src, "rb") as a, open(out, "rb") as b:
    same_data = a.read()[-16:] == b.read()[-16:]
check("--fix-ollama widens rope.dimension_sections and keeps the tensor data",
      r.returncode == 0 and "dimension_sections = [11, 11, 10, 0]" in shown and same_data)

r = run(src, src, "--fix-ollama")
check("writing over the input is refused, and the input is left intact",
      r.returncode != 0 and digest(src) == before)

link = os.path.join(TMP, "link.gguf")
os.symlink(src, link)
r = run(src, link, "--fix-ollama")
check("so is writing over it through another path", r.returncode != 0 and digest(src) == before)

subprocess.run(["rm", "-rf", TMP])
print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
