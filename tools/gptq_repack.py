#!/usr/bin/env python3
"""Repack the Qwen3.8-Flash-Next AutoRound W4A16 checkpoint's int4 tensors
into the engine's packed-int form, and verify the conversion against the
checkpoint's own layout and an independent quantization of the same weights.

THE TWO LAYOUTS. auto-round exports `packing_format: auto_round:auto_gptq`,
the auto_gptq v1 layout, as a triple per quantized Linear:

  qweight  I32 [K/8, N]      code(n, k) at word k/8, bits 4*(k%8)
  scales   F16 [K/group, N]  one per group of `group` elements along K
  qzeros   I32 [K/group, N/8]  the zero point minus one, 4 bits each

and dequant is w(n, k) = (code - (qzero + 1)) * scale. The engine's
packed-int form (models/quant_matrix.hpp, GlmPackedMatrix) is

  weight_packed I32 [N, K*bits/32]  code(n, k) at word k/8, bits 4*(k%8)
  weight_scale  BF16 [N, K/64]      one per group of 64 along K
  weight_shape  I64 [2]             [N, K]

and dequant is w(n, k) = (code - 2^(bits-1)) * scale.

So the two agree on everything inside a 32-bit word: both pack 8 consecutive
k into one word, lowest k in the low nibble. THE REPACK IS A WORD-LEVEL
TRANSPOSE — no bit manipulation at all — plus:

  * the zero point. sym=True writes qzeros = 7 everywhere, i.e. a zero point
    of 8 under auto_gptq's minus-one storage, which is exactly the engine's
    2^(bits-1) offset. The tool CHECKS this rather than assuming it, and
    refuses a checkpoint whose qzeros say anything else.
  * the group. The checkpoint's 128 becomes two of the engine's 64-element
    groups carrying the same scale. Exact, and it costs K/64 - K/128 extra
    bf16 per row.
  * the scale dtype. F16 (10 mantissa bits) narrowed to BF16 (7). This is
    the one lossy step in the conversion; `verify` measures it.

Usage:
  tools/gptq_repack.py inspect [model_dir]
  tools/gptq_repack.py verify  [model_dir] [--matrices N] [--reference ID]
  tools/gptq_repack.py repack  [model_dir] --out FILE [--layers A,B] [--experts N]

`verify` with --reference cross-checks the dequantized weights against an
independent NVFP4 quantization of the same model (nvidia/Qwen3.8-Flash-Next-NVFP4
by default): the two disagree by quantization noise if the layout reading is
right, and are uncorrelated if the bit order or the zero point is wrong.

Python 3 standard library only (no numpy, no torch): the tool must run on a
bare node. The full-checkpoint repack belongs in the loader (C++); this tool
converts a selectable subset, for fixtures and for checking the arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from array import array

TOOL_VERSION = "1"
DEFAULT_MODEL = "azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound"
DEFAULT_REFERENCE = "nvidia/Qwen3.8-Flash-Next-NVFP4"
ENGINE_GROUP = 64  # models/quant_matrix.hpp, kPackedGroup
WORDS_PER_I32 = 8  # 4-bit codes


# ---- the Hub cache (mirrors loaders/hf_cache.cpp) -----------------------
def cache_root():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def resolve_model(spec):
    """A directory, or a Hub model id resolved in the local cache."""
    if os.path.isdir(spec):
        return spec
    mroot = os.path.join(cache_root(), "models--" + spec.replace("/", "--"))
    if not os.path.isdir(mroot):
        sys.exit(f"no directory and no cached model '{spec}' under {cache_root()}")
    ref = os.path.join(mroot, "refs", "main")
    if os.path.isfile(ref):
        rev = open(ref).read().strip()
        snap = os.path.join(mroot, "snapshots", rev)
        if os.path.isdir(snap):
            return snap
    snaps = os.listdir(os.path.join(mroot, "snapshots"))
    if len(snaps) != 1:
        sys.exit(f"{spec}: no refs/main and {len(snaps)} snapshots — ambiguous")
    return os.path.join(mroot, "snapshots", snaps[0])


# ---- safetensors ---------------------------------------------------------
class Shards:
    """Every tensor of a snapshot, by name, read through its shard header."""

    def __init__(self, root, subdir=None):
        self.root = os.path.join(root, subdir) if subdir else root
        self.index = {}  # name -> (path, dtype, shape, begin, end)
        for fn in sorted(os.listdir(self.root)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(self.root, fn)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for name, v in hdr.items():
                if name == "__metadata__":
                    continue
                o = v["data_offsets"]
                self.index[name] = (path, v["dtype"], v["shape"], 8 + n + o[0], 8 + n + o[1])

    def __contains__(self, name):
        return name in self.index

    def dtype(self, name):
        return self.index[name][1]

    def shape(self, name):
        return self.index[name][2]

    def raw(self, name):
        path, _, _, begin, end = self.index[name]
        with open(path, "rb") as fh:
            fh.seek(begin)
            return fh.read(end - begin)

    def words(self, name):
        a = array("I")
        a.frombytes(self.raw(name))
        return a

    def halves(self, name):
        a = array("H")
        a.frombytes(self.raw(name))
        return a


# ---- scalar conversions (mirrors common/dtypes.hpp) ----------------------
def f16_to_float(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def bf16_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def float_to_bf16(f):
    """Round-to-nearest-even, the engine's float_to_bf16_bits."""
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    if (u & 0x7FFFFFFF) > 0x7F800000:
        return ((u >> 16) & 0x8000) | 0x7FC0
    u += 0x7FFF + ((u >> 16) & 1)
    return (u >> 16) & 0xFFFF


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_to_float(code):
    v = E2M1[code & 0x7]
    return -v if code & 0x8 else v


def e4m3_to_float(v):
    sign = -1.0 if v & 0x80 else 1.0
    exp, man = (v >> 3) & 0xF, v & 0x7
    if exp == 0xF and man == 0x7:
        return float("nan")
    if exp == 0:
        return sign * man * 0.125 * 0.015625
    return sign * (1.0 + man * 0.125) * 2.0 ** (exp - 7)


# ---- the checkpoint's contract -------------------------------------------
def read_quant_config(root):
    cfg = json.load(open(os.path.join(root, "config.json")))
    q = cfg.get("quantization_config")
    if q is None:
        sys.exit("config.json has no quantization_config")
    return cfg, q


def check_contract(q):
    """The one form this tool converts; anything else is refused by name."""
    def need(field, want):
        got = q.get(field)
        if got != want:
            sys.exit(f"quantization_config.{field}: expected {want!r}, got {got!r}")
    if q.get("quant_method") not in ("gptq", "auto-round"):
        sys.exit(f"quantization_config.quant_method: expected gptq/auto-round, "
                 f"got {q.get('quant_method')!r}")
    need("bits", 4)
    need("sym", True)
    need("desc_act", False)
    group = q.get("group_size")
    if group % ENGINE_GROUP != 0:
        sys.exit(f"quantization_config.group_size: {group} is not a multiple of the "
                 f"engine's {ENGINE_GROUP}-element group")
    return group


def quantized_bases(shards):
    """Every module with a qweight, as base name -> (N, K)."""
    out = {}
    for name in shards.index:
        if not name.endswith(".qweight"):
            continue
        base = name[: -len(".qweight")]
        kw, n = shards.shape(name)
        out[base] = (n, kw * WORDS_PER_I32)
    return out


# ---- the repack ----------------------------------------------------------
def check_qzeros(shards, base, n, k, group):
    """sym=True must mean a constant zero point of 2^(bits-1)."""
    zname = base + ".qzeros"
    if zname not in shards:
        return  # a release that omits qzeros under sym is already saying 8
    z = shards.words(zname)
    seen = set()
    for w in z:
        for s in range(0, 32, 4):
            seen.add((w >> s) & 0xF)
    if seen != {7}:
        sys.exit(f"{base}.qzeros: nibbles {sorted(seen)} — this tool converts only the "
                 f"symmetric form, whose zero point is 8 (stored as 7 under auto_gptq's "
                 f"minus-one convention) and matches the engine's 2^(bits-1) offset")


def repack_words(qweight, n, k):
    """[K/8, N] word-major-by-k -> [N, K/8] word-major-by-n. A transpose:
    the 32-bit words themselves are already the engine's."""
    kw = k // WORDS_PER_I32
    out = array("I", bytes(4 * n * kw))
    for row in range(n):
        out[row * kw : (row + 1) * kw] = qweight[row::n]  # the column, C-speed
    return out


def repack_scales(scales_f16, n, k, group):
    """F16 [K/group, N] -> BF16 [N, K/64], each checkpoint scale covering
    group/64 of the engine's groups."""
    per = group // ENGINE_GROUP
    ng_ckpt = k // group
    ng_out = k // ENGINE_GROUP
    out = array("H", bytes(2 * n * ng_out))
    for row in range(n):
        col = scales_f16[row::n]  # this row's K/group scales, ascending in k
        base = row * ng_out
        for g in range(ng_ckpt):
            b = float_to_bf16(f16_to_float(col[g]))
            for r in range(per):
                out[base + g * per + r] = b
    return out


def gptq_dequant(qweight, scales_f16, n, k, group, row, cols):
    """The checkpoint's own arithmetic, straight from the layout description:
    w(n, k) = (code - 8) * f16_scale. Independent of repack_*()."""
    vals = []
    for c in cols:
        word = qweight[(c // WORDS_PER_I32) * n + row]
        code = (word >> (4 * (c % WORDS_PER_I32))) & 0xF
        s = f16_to_float(scales_f16[(c // group) * n + row])
        vals.append((code - 8) * s)
    return vals


def packed_dequant(words, scales_bf16, n, k, row, cols):
    """The engine's arithmetic on the repacked bytes (packq_decode)."""
    kw = k // WORDS_PER_I32
    ng = k // ENGINE_GROUP
    vals = []
    for c in cols:
        word = words[row * kw + c // WORDS_PER_I32]
        code = (word >> (4 * (c % WORDS_PER_I32))) & 0xF
        s = bf16_to_float(scales_bf16[row * ng + c // ENGINE_GROUP])
        vals.append((code - 8) * s)
    return vals


# ---- the NVFP4 cross-check ----------------------------------------------
def reference_row(ref, base, row, cols):
    """One row of the reference's weights, whatever form it keeps them in:
    a plain BF16 tensor, or an NVFP4 triple. Returns (values, K, what) or
    None when the reference does not carry this module."""
    wname = base + ".weight"
    if wname not in ref:
        return None
    if ref.dtype(wname) in ("BF16", "F16"):
        n, k = ref.shape(wname)
        raw = ref.raw(wname)
        conv = bf16_to_float if ref.dtype(wname) == "BF16" else f16_to_float
        vals = [conv(struct.unpack_from("<H", raw, (row * k + c) * 2)[0]) for c in cols]
        return vals, k, ref.dtype(wname)
    if (base + ".weight_scale") not in ref:
        return None
    vals, k = nvfp4_dequant(ref, base, row, cols)
    return vals, k, "NVFP4"


def nvfp4_dequant(ref, base, row, cols):
    """e2m1(code) * e4m3(block scale) * the per-tensor scale, for one row."""
    payload = ref.raw(base + ".weight")
    scales = ref.raw(base + ".weight_scale")
    ws2_name = base + ".weight_scale_2"
    ws2 = struct.unpack("<f", ref.raw(ws2_name)[:4])[0] if ws2_name in ref else 1.0
    n, half = ref.shape(base + ".weight")
    k = half * 2
    sc_per_row = ref.shape(base + ".weight_scale")[1]
    group = k // sc_per_row
    vals = []
    for c in cols:
        byte = payload[row * half + c // 2]
        code = (byte >> 4) if (c & 1) else (byte & 0xF)
        s = e4m3_to_float(scales[row * sc_per_row + c // group])
        vals.append(e2m1_to_float(code) * s * ws2)
    return vals, k


def correlate(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((y - mb) ** 2 for y in b)
    if sa <= 0 or sb <= 0:
        return 0.0, 0.0
    ratio = sab / sa if sa > 0 else 0.0
    return sab / (sa * sb) ** 0.5, ratio


# ---- safetensors writing -------------------------------------------------
def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, bytes), written in name order."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        dt, shape, buf = tensors[name]
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [offset, offset + len(buf)]}
        offset += len(buf)
        blobs.append(buf)
    header["__metadata__"] = {"format": "pt", "producer": f"dgpp/gptq_repack.py v{TOOL_VERSION}"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for b in blobs:
            fh.write(b)


# ---- commands ------------------------------------------------------------
def classify(base):
    if ".mlp.experts." in base:
        return "routed expert"
    if base.endswith("lm_head"):
        return "lm_head"
    return "other"


def cmd_inspect(args):
    root = resolve_model(args.model)
    cfg, q = read_quant_config(root)
    group = check_contract(q)
    print(f"model      {root}")
    print(f"contract   {q['quant_method']} int{q['bits']} group {group} sym={q['sym']} "
          f"desc_act={q['desc_act']}")
    print(f"           -> engine packed-int, {group // ENGINE_GROUP} group(s) of "
          f"{ENGINE_GROUP} per checkpoint scale")
    shards = Shards(root)
    bases = quantized_bases(shards)
    kinds = {}
    for base, (n, k) in bases.items():
        kinds.setdefault(classify(base), []).append((base, n, k))
    total = 0
    for kind in sorted(kinds):
        entries = kinds[kind]
        shapes = sorted({(n, k) for _, n, k in entries})
        bytes_ = sum(n * k // 2 for _, n, k in entries)
        total += bytes_
        print(f"  {kind:>14}: {len(entries):6d} matrices  shapes(N,K)={shapes}  "
              f"{bytes_ / 2**30:.2f} GiB of codes")
    print(f"  {'total':>14}: {total / 2**30:.2f} GiB of int4 codes")
    dense = [n for n in shards.index if n.endswith(".weight_scale_inv")]
    print(f"dense      {len(dense)} block-FP8 tensors left alone "
          f"(attention, GDN, shared experts)")
    if "other" in kinds:
        print("WARNING: quantized modules outside the routed experts and the head:")
        for base, n, k in kinds["other"][:8]:
            print(f"  {base}  [{n}, {k}]")


def sample_bases(shards, limit):
    bases = quantized_bases(shards)
    experts = sorted(b for b in bases if classify(b) == "routed expert")
    picks = []
    # Spread over layers and over the three projections.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        matches = [b for b in experts if b.endswith(proj)]
        step = max(1, len(matches) // max(1, limit // 3))
        picks += matches[::step][: max(1, limit // 3)]
    heads = [b for b in bases if classify(b) == "lm_head"]
    return picks[:limit] + heads, bases


def cmd_verify(args):
    root = resolve_model(args.model)
    _, q = read_quant_config(root)
    group = check_contract(q)
    shards = Shards(root)
    picks, bases = sample_bases(shards, args.matrices)
    ref = None
    if args.reference:
        ref = Shards(resolve_model(args.reference))
        print(f"reference  {args.reference}")
    print(f"verifying {len(picks)} matrices from {root}\n")

    worst_scale_rel, worst_scale_where = 0.0, ""
    worst_w_rel, worst_w_where = 0.0, ""
    narrow_sq, narrow_n = 0.0, 0
    for base in picks:
        n, k = bases[base]
        check_qzeros(shards, base, n, k, group)
        qw = shards.words(base + ".qweight")
        sc = shards.halves(base + ".scales")
        if len(qw) != n * k // WORDS_PER_I32:
            sys.exit(f"{base}.qweight: {len(qw)} words for [{n}, {k}]")
        if len(sc) != n * k // group:
            sys.exit(f"{base}.scales: {len(sc)} values for [{n}, {k}] group {group}")
        words = repack_words(qw, n, k)
        scales = repack_scales(sc, n, k, group)

        # 1. The repack preserves the codes: the engine's arithmetic on the
        #    repacked bytes equals the checkpoint's own, exactly, once the
        #    scale is the same number. Sampled rows, every column.
        rows = [0, n // 3, n // 2, n - 1]
        cols = list(range(k))
        exact_bad = 0
        for row in rows:
            a = gptq_dequant(qw, sc, n, k, group, row, cols)
            b = packed_dequant(words, scales, n, k, row, cols)
            for ca, cb in zip(a, b):
                # The codes must agree bit for bit; the scales differ only by
                # the bf16 narrowing, so compare the relative gap.
                if ca == 0.0 or cb == 0.0:
                    exact_bad += (ca != 0.0) or (cb != 0.0)
                    continue
                rel = abs(ca - cb) / abs(ca)
                if (ca > 0) != (cb > 0) or rel > 2.0 ** -8:
                    exact_bad += 1
                if rel > worst_w_rel:
                    worst_w_rel, worst_w_where = rel, base
        if exact_bad:
            sys.exit(f"{base}: {exact_bad} elements disagree beyond the bf16 scale "
                     f"narrowing — the repack is wrong")

        # 2. The narrowing itself, over every scale in the matrix.
        for i in range(0, len(sc), max(1, len(sc) // 4096)):
            f = f16_to_float(sc[i])
            if f == 0.0:
                continue
            rel = abs(bf16_to_float(float_to_bf16(f)) - f) / abs(f)
            narrow_sq += rel * rel
            narrow_n += 1
            if rel > worst_scale_rel:
                worst_scale_rel, worst_scale_where = rel, base

        # 3. The independent quantization, if we have one.
        line = f"  [ok] {base}  [{n}, {k}]"
        got = reference_row(ref, base, 0, list(range(min(k, 4096)))) if ref else None
        if got is not None:
            rvals, rk, what = got
            if rk != k:
                line += f"  (reference K={rk} != {k}, skipped)"
            else:
                mine = gptq_dequant(qw, sc, n, k, group, 0, list(range(min(k, 4096))))
                r, ratio = correlate(mine, rvals)
                line += f"  vs {what}: r={r:.4f} slope={ratio:.4f}"
                if what in ("BF16", "F16"):
                    # An exact reference: report the quantization error itself.
                    num = sum((a - b) ** 2 for a, b in zip(mine, rvals))
                    den = sum(b * b for b in rvals)
                    line += f" rel_l2={(num / den) ** 0.5 * 100:.2f}%"
                if r < 0.95:
                    line += "  <-- LAYOUT MISMATCH"
        print(line)

    rms = (narrow_sq / narrow_n) ** 0.5 if narrow_n else 0.0
    print(f"\nrepack preserves the codes on every sampled row.")
    print(f"rms   bf16 scale narrowing : {rms * 100:.4f}%  (over {narrow_n} scales)")
    print(f"worst bf16 scale narrowing : {worst_scale_rel * 100:.4f}%  ({worst_scale_where})")
    print(f"worst weight delta from it : {worst_w_rel * 100:.4f}%  ({worst_w_where})")
    print(f"the bf16 bound is 2^-8 = {2.0 ** -8 * 100:.4f}%")


def cmd_repack(args):
    root = resolve_model(args.model)
    _, q = read_quant_config(root)
    group = check_contract(q)
    shards = Shards(root)
    bases = quantized_bases(shards)
    want = sorted(bases)
    if args.layers:
        keep = set(args.layers.split(","))
        want = [b for b in want if any(f".layers.{l}." in b for l in keep)
                or classify(b) == "lm_head"]
    if args.experts is not None:
        want = [b for b in want if classify(b) != "routed expert"
                or int(b.split(".mlp.experts.")[1].split(".")[0]) < args.experts]
    if not args.head:
        want = [b for b in want if classify(b) != "lm_head"]
    if not want:
        sys.exit("the selection is empty")
    out = {}
    done = 0
    for base in want:
        n, k = bases[base]
        check_qzeros(shards, base, n, k, group)
        words = repack_words(shards.words(base + ".qweight"), n, k)
        scales = repack_scales(shards.halves(base + ".scales"), n, k, group)
        out[base + ".weight_packed"] = ("I32", [n, k // WORDS_PER_I32], words.tobytes())
        out[base + ".weight_scale"] = ("BF16", [n, k // ENGINE_GROUP], scales.tobytes())
        out[base + ".weight_shape"] = ("I64", [2], struct.pack("<qq", n, k))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(want)}", flush=True)
    write_safetensors(args.out, out)
    size = os.path.getsize(args.out)
    print(f"wrote {len(want)} matrices ({size / 2**20:.1f} MiB) to {args.out}")

    # Read the file back through the same reader the engine's shape checks
    # describe, and put its rows against the checkpoint's own arithmetic.
    back = Shards(os.path.dirname(os.path.abspath(args.out)) or ".")
    checked = 0
    for base in want:
        n, k = bases[base]
        if back.shape(base + ".weight_packed") != [n, k // WORDS_PER_I32]:
            sys.exit(f"{base}.weight_packed: wrong shape on read-back")
        if back.shape(base + ".weight_scale") != [n, k // ENGINE_GROUP]:
            sys.exit(f"{base}.weight_scale: wrong shape on read-back")
        rec = struct.unpack("<qq", back.raw(base + ".weight_shape"))
        if list(rec) != [n, k]:
            sys.exit(f"{base}.weight_shape: {rec} != [{n}, {k}]")
        words = back.words(base + ".weight_packed")
        scales = back.halves(base + ".weight_scale")
        qw = shards.words(base + ".qweight")
        sc = shards.halves(base + ".scales")
        for row in (0, n // 2, n - 1):
            a = gptq_dequant(qw, sc, n, k, group, row, range(k))
            b = packed_dequant(words, scales, n, k, row, range(k))
            for ca, cb in zip(a, b):
                if (ca == 0.0) != (cb == 0.0) or (ca != 0.0 and
                                                  abs(ca - cb) / abs(ca) > 2.0 ** -8):
                    sys.exit(f"{base}: the written bytes disagree with the checkpoint")
            checked += 1
    print(f"read back and checked {checked} rows against the source")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "verify", "repack"):
        p = sub.add_parser(name)
        p.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                       help="a directory or a cached Hub model id")
        if name == "verify":
            p.add_argument("--matrices", type=int, default=6)
            p.add_argument("--reference", default=DEFAULT_REFERENCE,
                           help="an independent quantization to cross-check against "
                                "('' to skip)")
        if name == "repack":
            p.add_argument("--out", required=True)
            p.add_argument("--layers", default="0", help="comma-separated layer indices")
            p.add_argument("--experts", type=int, default=2,
                           help="experts per layer (from 0)")
            p.add_argument("--head", action="store_true", help="include the lm_head")
    args = ap.parse_args()
    {"inspect": cmd_inspect, "verify": cmd_verify, "repack": cmd_repack}[args.cmd](args)


if __name__ == "__main__":
    main()
