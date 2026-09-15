#!/usr/bin/env python3
"""patch-gguf.py — rewrite GGUF metadata keys, streaming the tensor data through.

Why this exists: ollama's GGUF writer and llama.cpp's reader disagree on a few
metadata keys. For qwen35moe, ollama writes `rope.dimension_sections` with 3
elements while llama.cpp's hparams hold `std::array<int,4>` and hard-fail with
"wrong array length; expected 4, got 3". Upstream GGUFs from unsloth carry
[11, 11, 10, 0] — the same values plus a zero — so an ollama blob only needs
that one key widened to load natively.

Only the key-value section is rebuilt; tensor info and tensor data are copied
byte for byte. Tensor offsets are relative to the start of the data section, so
they stay valid as long as the data section keeps its alignment, which it does.

Usage:
  patch-gguf.py IN.gguf OUT.gguf --set KEY=JSON [--set KEY=JSON ...]
  patch-gguf.py IN.gguf OUT.gguf --fix-ollama      # apply known ollama fixes
  patch-gguf.py IN.gguf --show                     # dump metadata, no output file
  patch-gguf.py IN.gguf --template                 # print the embedded chat template

The value's GGUF type is taken from the existing key, so --set can only change
values, never introduce or retype a key.
"""
import json
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
# GGUF value type ids -> struct format / byte width
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_STR, TYPE_ARRAY = 8, 9

# Known ollama -> llama.cpp metadata fixes, keyed by general.architecture.
OLLAMA_FIXES = {
    "qwen35moe": {
        # llama.cpp reads mrope sections into std::array<int,4>; ollama emits 3.
        "rope.dimension_sections": lambda v: list(v) + [0] if len(v) == 3 else None,
    },
}


class Reader:
    def __init__(self, f):
        self.f = f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == TYPE_STR:
            return self.string()
        if t == TYPE_ARRAY:
            et, n = self.u32(), self.u64()
            if et == TYPE_STR:
                return (et, [self.string() for _ in range(n)])
            if et == TYPE_ARRAY:
                return (et, [self.value(TYPE_ARRAY) for _ in range(n)])
            return (et, list(struct.unpack("<" + FMT[et] * n, self.f.read(SIZE[et] * n))))
        return struct.unpack("<" + FMT[t], self.f.read(SIZE[t]))[0]


def wr_u32(v):
    return struct.pack("<I", v)


def wr_u64(v):
    return struct.pack("<Q", v)


def wr_str(s):
    b = s.encode("utf-8")
    return wr_u64(len(b)) + b


def wr_value(t, v):
    if t == TYPE_STR:
        return wr_str(v)
    if t == TYPE_ARRAY:
        et, items = v
        out = wr_u32(et) + wr_u64(len(items))
        if et == TYPE_STR:
            return out + b"".join(wr_str(s) for s in items)
        if et == TYPE_ARRAY:
            return out + b"".join(wr_value(TYPE_ARRAY, i) for i in items)
        return out + struct.pack("<" + FMT[et] * len(items), *items)
    return struct.pack("<" + FMT[t], v)


def read_header(path):
    """Return (kv list, n_tensors, kv_end, data_start, alignment)."""
    f = open(path, "rb")
    r = Reader(f)
    if f.read(4) != GGUF_MAGIC:
        sys.exit(f"{path}: not a GGUF file")
    version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
    if version != 3:
        print(f"warning: GGUF version {version}, tested against v3", file=sys.stderr)
    kv = []
    for _ in range(n_kv):
        k = r.string()
        t = r.u32()
        kv.append([k, t, r.value(t)])
    kv_end = f.tell()
    # tensor info: name, n_dims, dims[], ggml_type, offset
    for _ in range(n_tensors):
        r.string()
        nd = r.u32()
        f.seek(8 * nd + 4 + 8, os.SEEK_CUR)
    ti_end = f.tell()
    align = next((v for k, _, v in kv if k == "general.alignment"), 32)
    data_start = ti_end + (-ti_end % align)
    f.close()
    return kv, version, n_tensors, kv_end, ti_end, data_start, align


def show(kv):
    for k, t, v in kv:
        if t == TYPE_ARRAY:
            et, items = v
            v = f"<array[{et}] len {len(items)}>" if len(items) > 12 else items
        elif t == TYPE_STR and len(v) > 100:
            v = f"<str {len(v)} chars>"
        print(f"  {k} = {v}")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src = sys.argv[1]
    kv, version, n_tensors, kv_end, ti_end, data_start, align = read_header(src)
    arch = next((v for k, _, v in kv if k == "general.architecture"), "")

    if "--template" in sys.argv:
        tpl = next((v for k, _, v in kv if k == "tokenizer.chat_template"), None)
        if tpl is None:
            sys.exit("no tokenizer.chat_template in this GGUF")
        sys.stdout.write(tpl)
        return

    if "--show" in sys.argv:
        print(f"{src}: arch={arch} tensors={n_tensors} kv={len(kv)} data_start={data_start}")
        show(kv)
        return

    dst = sys.argv[2]
    edits = {}
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--set":
            k, _, raw = args[i + 1].partition("=")
            edits[k] = json.loads(raw)
            i += 2
        elif args[i] == "--fix-ollama":
            for suffix, fix in OLLAMA_FIXES.get(arch, {}).items():
                key = f"{arch}.{suffix}"
                cur = next((v for k, t, v in kv if k == key), None)
                if cur is None:
                    print(f"note: {key} absent, nothing to fix")
                    continue
                new = fix(cur[1] if isinstance(cur, tuple) else cur)
                if new is None:
                    print(f"note: {key} already correct")
                else:
                    edits[key] = new
            i += 1
        else:
            sys.exit(f"unknown argument: {args[i]}")

    if not edits:
        sys.exit("nothing to patch (no --set given and no ollama fix applied)")

    for k, t, v in kv:
        if k not in edits:
            continue
        new = edits.pop(k)
        if t == TYPE_ARRAY:
            et, old = v
            print(f"  {k}: {old} -> {new}")
            kv[kv.index([k, t, v])][2] = (et, new)
        else:
            print(f"  {k}: {v} -> {new}")
            kv[kv.index([k, t, v])][2] = new
    if edits:
        sys.exit(f"key(s) not present in file, refusing to add: {list(edits)}")

    head = GGUF_MAGIC + wr_u32(version) + wr_u64(n_tensors) + wr_u64(len(kv))
    head += b"".join(wr_str(k) + wr_u32(t) + wr_value(t, v) for k, t, v in kv)

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(head)
        fin.seek(kv_end)
        fout.write(fin.read(ti_end - kv_end))          # tensor info, verbatim
        pad = -fout.tell() % align
        fout.write(b"\0" * pad)
        assert fout.tell() % align == 0
        fin.seek(data_start)
        copied = 0
        while True:
            chunk = fin.read(64 << 20)
            if not chunk:
                break
            fout.write(chunk)
            copied += len(chunk)
            print(f"\r  copying tensor data: {copied / 1e9:.1f} GB", end="", flush=True)
    print(f"\nwrote {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
