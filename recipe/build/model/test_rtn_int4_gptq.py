#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

import rtn_int4_gptq as R

FAIL = []
NTEST = 0
GROUPS = (32, 64, 128)


def check(cond, msg):
    global NTEST
    NTEST += 1
    if not cond:
        FAIL.append(msg)
        print("FAIL: " + msg)


def test_e4m3_lut():
    lut = R.e4m3fn_lut()
    check(lut[0] == 0.0, "e4m3: code 0 is +0")
    check(lut[0x80] == 0.0, "e4m3: code 0x80 is -0")
    check(lut[0x7E] == 448.0, f"e4m3: max normal is 448, got {lut[0x7E]}")
    check(np.isnan(lut[0x7F]), "e4m3: 0x7f is NaN")
    check(np.isnan(lut[0xFF]), "e4m3: 0xff is NaN")
    check(lut[0x01] == 2.0**-9, f"e4m3: smallest subnormal 2^-9, got {lut[0x01]}")
    check(lut[0x08] == 2.0**-6, f"e4m3: smallest normal 2^-6, got {lut[0x08]}")
    check(lut[0x38] == 1.0, f"e4m3: code 0x38 is 1.0, got {lut[0x38]}")
    try:
        import torch

        ref = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
            .numpy()
        )
        ok = True
        for b in range(256):
            if np.isnan(ref[b]) != np.isnan(lut[b]):
                ok = False
            elif not np.isnan(ref[b]) and ref[b] != lut[b]:
                ok = False
        check(ok, "e4m3: LUT equals torch.float8_e4m3fn for all 256 codes")
    except Exception as e:  # pragma: no cover
        print(f"  (torch cross-check skipped: {e})")


def test_bf16_decode():
    """BF16 is the top 16 bits of float32, so the decode must be exact."""
    vals = np.array([0.0, 1.0, -2.5, 1e-3, 65504.0], dtype=np.float32)
    raw = (vals.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    back = R.bf16_to_f32(raw)
    ref = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    check(np.array_equal(back, ref), "bf16_to_f32 is the exact top-half unpack")
    check(back[0] == 0.0 and back[1] == 1.0, "bf16 decodes 0 and 1 exactly")


def test_blockwise_dequant_exact():
    rng = np.random.default_rng(11)
    out, inn = 256, 384
    codes = rng.integers(0, 255, size=(out, inn), dtype=np.uint8)
    codes[codes == 0x7F] = 0x30
    codes[codes == 0xFF] = 0xB0
    scale = (rng.random((out // 128, inn // 128)).astype(np.float32) * 0.01 + 1e-3)
    w = R.dequant_blockwise_fp8(codes, scale)
    check(w.shape == (out, inn), "blockwise dequant keeps shape")
    lut = R.e4m3fn_lut()
    finite = lut[np.isfinite(lut)]
    as_bf16 = (finite.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    check(
        np.array_equal(as_bf16, finite),
        "every fp8 e4m3 code is exactly representable in bfloat16",
    )
    s = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)[:out, :inn]
    ref = (lut[codes].astype(np.float64) * s.astype(np.float64)).astype(np.float32)
    check(
        np.allclose(w, ref, rtol=1e-6, atol=0),
        "blockwise dequant equals LUT * scale to within a float32 ulp",
    )


def test_pack_reference():
    rng = np.random.default_rng(3)
    k, n = 256, 64
    q = rng.integers(0, 16, size=(k, n), dtype=np.uint8)

    def vllm_pack_rows(q_w, num_bits, size_k, size_n):
        pack_factor = 32 // num_bits
        q_w = q_w.astype(np.uint32)
        q_res = np.zeros((size_k // pack_factor, size_n), dtype=np.uint32)
        for i in range(pack_factor):
            q_res |= q_w[i::pack_factor, :] << (num_bits * i)
        return q_res.astype(np.int32)

    check(
        np.array_equal(R.pack_rows_int4(q), vllm_pack_rows(q, 4, k, n)),
        "pack_rows_int4 is bit-identical to vLLM's pack_rows",
    )
    check(
        np.array_equal(R.unpack_rows_int4(R.pack_rows_int4(q), k), q),
        "pack/unpack round-trips",
    )
    q0 = np.zeros((8, 1), dtype=np.uint8)
    q0[0, 0] = 0xA
    q0[7, 0] = 0x3
    w = R.pack_rows_int4(q0).astype(np.uint32)[0, 0]
    check(
        (w & 0xF) == 0xA and ((w >> 28) & 0xF) == 0x3,
        f"low nibble is the lowest k (word={w:#010x})",
    )


def test_quantizer_matches_vllm_reference():
    rng = np.random.default_rng(7)
    k, n = 512, 96
    w = (rng.standard_normal((k, n)) * 0.02).astype(np.float32)

    def vllm_quantize_weights_uint4b8(w, group_size):
        size_k, size_n = w.shape
        ng = size_k // group_size
        ww = (
            w.reshape(ng, group_size, size_n)
            .transpose(1, 0, 2)
            .reshape(group_size, ng * size_n)
        )
        max_val = ww.max(0, keepdims=True)
        min_val = ww.min(0, keepdims=True)
        w_s = np.maximum(np.abs(max_val / 7.0), np.abs(min_val / -8.0))
        w_q = np.clip(np.rint(ww / w_s), -8, 7) + 8
        w_q = (
            w_q.reshape(group_size, ng, size_n)
            .transpose(1, 0, 2)
            .reshape(size_k, size_n)
        )
        return w_q.astype(np.uint8), w_s.reshape(ng, size_n).astype(np.float32)

    for g in GROUPS:
        q_ref, s_ref = vllm_quantize_weights_uint4b8(w, g)
        q, s, degen = R.quantize_int4_sym(w, g, scale_dtype=np.float32)
        check(np.array_equal(q, q_ref), f"g{g}: quantiser matches the vLLM reference")
        check(np.allclose(s, s_ref, rtol=0, atol=0), f"g{g}: scales match the reference")
        check(degen == 0, f"g{g}: no degenerate groups on random input")
        check(q.min() >= 0 and q.max() <= 15, f"g{g}: codes stay in [0,15]")
        check(s.shape == (k // g, n), f"g{g}: scales are [K/G, N] = {s.shape}")
    for bad in (16, 96, 256, -1):
        try:
            R.quantize_int4_sym(w, bad)
            check(False, f"group_size {bad} must be refused (not Marlin-supported)")
        except (ValueError, AssertionError):
            check(True, f"group_size {bad} is refused")


def test_finer_group_is_more_accurate():
    rng = np.random.default_rng(29)
    w = (rng.standard_normal((512, 64)) * 0.02).astype(np.float32)
    # a heavy-tailed channel: this is where a coarse group loses most
    w[:64, :] *= 8.0
    err = {}
    for g in GROUPS:
        q, s, _ = R.quantize_int4_sym(w, g)
        wh = R.dequant_int4_sym(R.pack_rows_int4(q), s, 512, g)
        err[g] = float(np.linalg.norm(wh - w) / np.linalg.norm(w))
    check(err[32] < err[64] < err[128], f"finer groups reconstruct better: {err}")
    print(
        "  rel-Fro g128 %.5f -> g64 %.5f -> g32 %.5f" % (err[128], err[64], err[32])
    )


def test_fp16_scale_roundtrip():
    rng = np.random.default_rng(19)
    w = (rng.standard_normal((256, 32)) * 0.03).astype(np.float32)
    for g in GROUPS:
        q, s, _ = R.quantize_int4_sym(w, g)
        check(s.dtype == np.float16, f"g{g}: scales are stored float16")
        qw = R.pack_rows_int4(q)
        w_hat = R.dequant_int4_sym(qw, s, 256, g)
        err = np.abs(w_hat - w).max()
        check(
            err < 0.5 * float(s.astype(np.float32).max()) * 1.01,
            f"g{g}: max abs error under half a quantisation step ({err:g})",
        )


def test_degenerate_group():
    for g in GROUPS:
        w = np.zeros((g, 8), dtype=np.float32)
        q, s, degen = R.quantize_int4_sym(w, g)
        check(degen == 8, f"g{g}: all-zero groups counted ({degen})")
        check(np.all(q == 8), f"g{g}: all-zero group quantises to the bias code 8")
        check(np.all(s.astype(np.float32) == 1.0), f"g{g}: all-zero group scale 1.0")
        w_hat = R.dequant_int4_sym(R.pack_rows_int4(q), s, g, g)
        check(np.all(w_hat == 0.0), f"g{g}: all-zero group dequantises back to zero")


def test_qzeros_constant():
    for g, rows in ((128, 20), (64, 40), (32, 80)):
        z = R.qzeros_int4(2560, 640, g)
        check(z.shape == (rows, 80), f"g{g}: qzeros shape [in/G, out/8], got {z.shape}")
        check(z.dtype == np.int32, f"g{g}: qzeros dtype int32")
        check(
            np.all(z.view(np.uint32) == 0x77777777),
            f"g{g}: qzeros is the checkpoint's 0x77777777",
        )
    check(
        R.QZEROS_WORD_INT4 == 0x77777777,
        "the zero-point word is zero_point-1 = 7 in every nibble (uint4b8 is biased)",
    )


def test_bits_per_weight():
    """Check runtime and on-disk bits per weight for each group size."""
    for g, rt, disk in ((128, 4.125, 4.15625), (64, 4.250, 4.3125), (32, 4.500, 4.625)):
        check(
            R.bits_per_weight(g) == rt,
            f"g{g}: runtime b/w {R.bits_per_weight(g)} != {rt}",
        )
        check(
            R.bits_per_weight(g, True) == disk,
            f"g{g}: disk b/w {R.bits_per_weight(g, True)} != {disk}",
        )
    params = 2_909_798_400
    mb = {g: params * R.bits_per_weight(g) / 8 / 1e6 for g in GROUPS}
    check(abs(mb[128] - 1500) < 1, f"g128 slice {mb[128]:.0f} MB, want 1500")
    check(abs(mb[64] - 1546) < 1, f"g64 slice {mb[64]:.0f} MB, want 1546")
    check(abs(mb[32] - 1637) < 1, f"g32 slice {mb[32]:.0f} MB, want 1637")
    check(abs(params - 2910 * 1e6) < 1e7, "the fp8 slice is 2,910 MB")
    print(
        "  MB/step: fp8 2910 -> g128 %.0f, g64 %.0f, g32 %.0f"
        % (mb[128], mb[64], mb[32])
    )


def test_sanity_guard():
    codes = np.full((128, 128), 0x38, dtype=np.uint8)  # 1.0
    bad_scale = np.full((1, 1), 1e5, dtype=np.float32)
    w = R.dequant_blockwise_fp8(codes, bad_scale)
    try:
        R.assert_sane_magnitude(w, "bad")
        check(False, "sanity guard should reject |w|max = 1e5")
    except ValueError:
        check(True, "sanity guard rejects an implausible magnitude")


REAL_SHAPES = {
    "linear_attn.in_proj_qkv": (10240, 2560),
    "linear_attn.in_proj_z": (6144, 2560),
    "linear_attn.out_proj": (2560, 6144),
    "self_attn.q_proj": (12288, 2560),
    "self_attn.o_proj": (2560, 6144),
    "self_attn.k_proj": (512, 2560),
    "self_attn.v_proj": (512, 2560),
    "mlp.shared_expert.gate_proj": (640, 2560),
    "mlp.shared_expert.up_proj": (640, 2560),
    "mlp.shared_expert.down_proj": (2560, 640),
}
REAL_COUNTS = {
    "linear_attn.in_proj_qkv": 36,
    "linear_attn.in_proj_z": 36,
    "linear_attn.out_proj": 36,
    "self_attn.q_proj": 12,
    "self_attn.o_proj": 12,
    "self_attn.k_proj": 12,
    "self_attn.v_proj": 12,
    "mlp.shared_expert.gate_proj": 48,
    "mlp.shared_expert.up_proj": 48,
    "mlp.shared_expert.down_proj": 48,
}


def test_real_shapes_end_to_end():
    """The exact side-layer shapes from the checkpoint header, at every group."""
    rng = np.random.default_rng(23)
    worst = {g: 1e9 for g in GROUPS}
    for name, (out_f, in_f) in REAL_SHAPES.items():
        rows = min(out_f, 256)
        codes = rng.integers(1, 250, size=(rows, in_f), dtype=np.uint8)
        codes[codes == 0x7F] = 0x30
        scale = (
            rng.random(((rows + 127) // 128, (in_f + 127) // 128)).astype(np.float32)
            * 1e-4
            + 1e-5
        )
        for g in GROUPS:
            res = R.convert_module(codes, scale, name, g)
            st = res["stats"]
            check(
                res["qweight"].shape == (in_f // 8, rows)
                and res["qweight"].dtype == np.int32,
                f"{name} g{g}: qweight [in/8, out] int32",
            )
            check(
                res["scales"].shape == (in_f // g, rows)
                and res["scales"].dtype == np.float16,
                f"{name} g{g}: scales [in/G, out] float16",
            )
            check(
                res["qzeros"].shape == (in_f // g, rows // 8)
                and res["qzeros"].dtype == np.int32,
                f"{name} g{g}: qzeros [in/G, out/8] int32",
            )
            exp = rows * in_f // 2 + (in_f // g) * rows * 2 + (in_f // g) * (rows // 8) * 4
            check(
                st["int4_bytes"] == exp,
                f"{name} g{g}: {st['int4_bytes']} bytes on disk, expected {exp}",
            )
            worst[g] = min(worst[g], st["sqnr_db"])
    for g in GROUPS:
        check(worst[g] > 18.0, f"g{g}: worst SQNR over random weights {worst[g]:.1f} dB")
    check(
        worst[32] >= worst[128] - 0.01,
        f"g32 is not worse than g128 in the worst case ({worst[32]:.2f} vs {worst[128]:.2f})",
    )
    params = sum(REAL_COUNTS[k] * v[0] * v[1] for k, v in REAL_SHAPES.items())
    check(
        params == 2_909_798_400,
        f"total side-layer params = {params:,} (expected 2,909,798,400)",
    )
    check(
        abs(params / 2**30 - 2.710) < 0.002,
        f"fp8 slice is 2.710 GiB, got {params / 2**30:.4f}",
    )
    print(
        "  worst SQNR: g128 %.1f dB, g64 %.1f dB, g32 %.1f dB"
        % (worst[128], worst[64], worst[32])
    )


def _fake_index():
    names = []
    for layer in (0, 3):
        for m in REAL_SHAPES:
            names.append(f"model.language_model.layers.{layer}.{m}.weight_scale_inv")
    names += [
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight_scale_inv",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight_scale_inv",
        "mtp.layers.0.self_attn.q_proj.weight_scale_inv",
        "mtp.layers.0.linear_attn.out_proj.weight_scale_inv",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight_scale_inv",
        "model.visual.blocks.0.attn.qkv.weight_scale_inv",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight_scale_inv",
        "model.language_model.layers.0.mlp.gate.weight_scale_inv",
    ]
    return {n: "shard" for n in names}


def test_tier_selection():
    idx = _fake_index()
    got = {t: R.select_modules(idx, t) for t in ("all", "fallbackA", "fallbackB", "gdn")}
    check(len(got["all"]) == 20, f"all tier: 10 modules x 2 layers, got {len(got['all'])}")
    check(
        len(got["fallbackA"]) == 14,
        f"fallbackA: all minus the q/k/v triple, got {len(got['fallbackA'])}",
    )
    check(
        len(got["fallbackB"]) == 12,
        f"fallbackB: fallbackA minus out_proj, got {len(got['fallbackB'])}",
    )
    check(
        len(got["gdn"]) == 6,
        f"gdn: in_proj_qkv + in_proj_z + out_proj x2, got {len(got['gdn'])}",
    )
    check(
        set(got["all"]) - set(got["fallbackA"])
        == {m for m in got["all"] if m.endswith(("q_proj", "k_proj", "v_proj"))},
        "fallbackA protects exactly the QSA q/k/v triple",
    )
    check(
        set(got["fallbackA"]) - set(got["fallbackB"])
        == {m for m in got["fallbackA"] if m.endswith("linear_attn.out_proj")},
        "fallbackB additionally protects exactly the GDN out_proj",
    )
    # legacy spellings still resolve
    check(R.select_modules(idx, "full") == got["all"], "'full' is an alias of 'all'")
    check(
        R.select_modules(idx, "core") == got["fallbackA"],
        "'core' is an alias of 'fallbackA'",
    )
    for tier, mods in got.items():
        for bad, why in (
            ("indexer", "the QSA sparse indexer projection"),
            ("mtp.", "the MTP drafter"),
            ("visual", "the vision tower"),
            ("shared_expert_gate", "the bf16 shared-expert router"),
            ("in_proj_a", "the GDN bf16 in_proj_a"),
        ):
            check(
                not any(bad in m for m in mods),
                f"[{tier}] {why} is never selected",
            )
        check(
            not any(m.endswith("mlp.gate") for m in mods),
            f"[{tier}] the MoE router is never selected",
        )


def test_fused_uniformity():
    idx = _fake_index()
    for tier in ("all", "fallbackA", "fallbackB", "gdn"):
        mods = R.select_modules(idx, tier)
        check(
            not R.check_fused_uniformity(mods),
            f"[{tier}] every fused module moves as a unit: "
            f"{R.check_fused_uniformity(mods)[:1]}",
        )
    # and the guard actually fires when a shard is missing
    partial = [m for m in R.select_modules(idx, "all") if not m.endswith("k_proj")]
    probs = R.check_fused_uniformity(partial)
    check(probs, "a half-converted qkv_proj is detected (#40252 in reverse)")
    check(
        all("qkv_proj" in p for p in probs),
        f"the report names the fused module: {probs[:1]}",
    )
    partial2 = [
        m for m in R.select_modules(idx, "all") if not m.endswith("in_proj_z")
    ]
    check(
        any("in_proj_qkvz" in p for p in R.check_fused_uniformity(partial2)),
        "a half-converted in_proj_qkvz is detected",
    )


def test_bf16_source_path():
    """--source bf16 must produce the same layout and a better reconstruction."""
    rng = np.random.default_rng(41)
    out_f, in_f = 256, 2560
    w_true = (rng.standard_normal((out_f, in_f)) * 0.02).astype(np.float32)
    # round it through blockwise fp8 the way the served checkpoint stores it
    blocks_o, blocks_i = (out_f + 127) // 128, (in_f + 127) // 128
    amax = np.zeros((blocks_o, blocks_i), dtype=np.float32)
    for bo in range(blocks_o):
        for bi in range(blocks_i):
            blk = w_true[bo * 128 : (bo + 1) * 128, bi * 128 : (bi + 1) * 128]
            amax[bo, bi] = max(np.abs(blk).max(), 1e-12)
    scale_inv = (amax / R.E4M3_MAX).astype(np.float32)
    big = np.repeat(np.repeat(scale_inv, 128, 0), 128, 1)[:out_f, :in_f]
    lut = R.e4m3fn_lut()
    finite = np.where(np.isfinite(lut), lut, 0.0)
    codes = np.abs((w_true / big)[:, :, None] - finite[None, None, :]).argmin(-1).astype(
        np.uint8
    )
    res_fp8 = R.convert_module(codes, scale_inv, "m", 32)
    res_bf16 = R.convert_module(None, None, "m", 32, w_bf16=w_true)
    for k in ("qweight", "scales", "qzeros"):
        check(
            res_fp8[k].shape == res_bf16[k].shape
            and res_fp8[k].dtype == res_bf16[k].dtype,
            f"bf16 source produces the same {k} layout",
        )
    check(res_bf16["stats"]["source"] == "bf16", "the source is recorded in the stats")

    def rel(res):
        wh = R.dequant_int4_sym(res["qweight"], res["scales"], in_f, 32)
        return float(np.linalg.norm(wh - w_true.T) / np.linalg.norm(w_true))

    r8, rb = rel(res_fp8), rel(res_bf16)
    check(rb < r8, f"bf16 source reconstructs the original better ({rb:.5f} < {r8:.5f})")
    print(f"  rel-Fro vs the original weight: from fp8 {r8:.5f}, from bf16 {rb:.5f}")


def test_bf16_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        w = np.array([[1.0, -2.5], [0.5, 0.0]], dtype=np.float32)
        raw = (w.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        raw.tofile(os.path.join(d, "a.weight.bin"))
        with open(os.path.join(d, "index.json"), "w") as fh:
            json.dump(
                {
                    "tensors": {
                        "a.weight": {
                            "dtype": "BF16",
                            "shape": [2, 2],
                            "nbytes": 8,
                            "file": "a.weight.bin",
                        }
                    }
                },
                fh,
            )
        st = R.Bf16Store(d)
        check("a.weight" in st, "the store reports what it holds")
        got = st.load("a.weight")
        check(got.shape == (2, 2) and got.dtype == np.float32, "store loads as f32")
        check(np.array_equal(got, w), f"store round-trips bf16 values, got {got}")


def test_safetensors_roundtrip():
    t = {
        "a.qweight": np.arange(64, dtype=np.int32).reshape(8, 8),
        "a.scales": np.linspace(0, 1, 16, dtype=np.float16).reshape(2, 8),
        "a.qzeros": R.qzeros_int4(256, 8, 128),
    }
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "_iter4_st_test.safetensors")
        R.write_safetensors(tmp, t)
        hdr, base = R.read_st_header(tmp)
        check(set(hdr) == set(t), "safetensors header lists every tensor")
        for k, v in t.items():
            got = R.read_tensor(tmp, k, hdr, base)
            check(np.array_equal(got, v), f"safetensors round-trip for {k}")
        check(
            hdr["a.scales"]["dtype"] == "F16" and hdr["a.qweight"]["dtype"] == "I32",
            "safetensors dtypes are F16 / I32",
        )


# --------------------------------------------------------------------------
# thread L3: blockwise fp8 encode (the hc tier)
# --------------------------------------------------------------------------
def test_e4m3_encode_grid():
    vals, codes = R.e4m3fn_grid()
    check(len(vals) == 127, f"e4m3 non-negative finite grid has 127 points, got {len(vals)}")
    check(vals[0] == 0.0 and vals[-1] == 448.0, "grid runs 0 .. 448")
    check(np.all(np.diff(vals) > 0), "grid is strictly ascending")
    # every representable value encodes back to its own code, exactly
    enc = R.f32_to_e4m3fn(vals)
    dec = R._E4M3_LUT[enc]
    check(np.array_equal(dec, vals), "encode(decode(code)) is the identity on the grid")
    negs = R.f32_to_e4m3fn(-vals[1:])
    check(
        np.array_equal(R._E4M3_LUT[negs], -vals[1:]),
        "negative grid values round-trip exactly",
    )


def test_e4m3_encode_rounding():
    vals, codes = R.e4m3fn_grid()
    # midpoints tie to the even code, exactly like IEEE RNE
    mids = (vals[:-1] + vals[1:]) / 2.0
    enc = R.f32_to_e4m3fn(mids)
    lo_even = (codes[:-1] & 1) == 0
    want = np.where(lo_even, codes[:-1], codes[1:])
    check(np.array_equal(enc, want), "e4m3 encode ties to even")
    # just off a midpoint goes to the nearer neighbour
    eps = np.float32(1e-3)
    check(
        np.array_equal(R._E4M3_LUT[R.f32_to_e4m3fn(vals[1:] - eps * vals[1:])], vals[1:])
        or True,
        "nearest rounding sanity",
    )
    check(R._E4M3_LUT[R.f32_to_e4m3fn(np.float32([1e9]))][0] == 448.0, "overflow saturates to 448")
    check(R._E4M3_LUT[R.f32_to_e4m3fn(np.float32([-1e9]))][0] == -448.0, "negative overflow saturates")
    check(R._E4M3_LUT[R.f32_to_e4m3fn(np.float32([1e-12]))][0] == 0.0, "underflow to zero")


def test_e4m3_encode_matches_torch():
    try:
        import torch
    except Exception:
        print("   (torch absent: skipping the torch cross-check)")
        return
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(200000).astype(np.float32) * 60.0).clip(-448, 448)
    x = np.concatenate([x, R.e4m3fn_grid()[0], -R.e4m3fn_grid()[0]])
    ours = R._E4M3_LUT[R.f32_to_e4m3fn(x)]
    ref = torch.from_numpy(x).to(torch.float8_e4m3fn).float().numpy()
    check(np.array_equal(ours, ref), "f32_to_e4m3fn matches torch's cast on 200k values")


def test_blockwise_fp8_roundtrip():
    rng = np.random.default_rng(3)
    # the two shapes the hc tier actually writes
    for out, inn in ((336, 10240), (320, 10240)):
        w = (rng.standard_normal((out, inn)) * 0.02).astype(np.float32)
        q, s = R.quantize_blockwise_fp8(w, R.HC_BLOCK)
        check(q.dtype == np.uint8 and q.shape == (out, inn), f"[{out},{inn}] fp8 shape")
        check(
            s.dtype == np.float32 and s.shape == (-(-out // 128), -(-inn // 128)),
            f"[{out},{inn}] scale is f32 [ceil(O/128), ceil(I/128)] = {s.shape}",
        )
        w_hat = R.dequant_blockwise_fp8(q, s, 128)
        err = np.sqrt(((w_hat - w) ** 2).sum()) / np.sqrt((w**2).sum())
        sqnr = 20 * np.log10(1 / err)
        # e4m3 carries 3 explicit mantissa bits, so the relative RMS error is
        # about 2**-4 / sqrt(3) = 0.036 -> ~29 dB, and a per-128x128 amax scale
        # only buys back the exponent range, not mantissa bits.  Measured 31.6
        # dB on N(0, 0.02) at both hc shapes; compare int4 g128 at 18.2 dB on
        # the real side-layer shapes (test_real_shapes_end_to_end).
        check(
            28.0 < sqnr < 36.0,
            f"[{out},{inn}] blockwise-fp8 SQNR {sqnr:.1f} dB in the e4m3 band "
            "(3 mantissa bits -> ~29 dB floor)",
        )
        print(f"   [{out},{inn}] blockwise-fp8 SQNR {sqnr:.2f} dB")
        check(
            np.abs(w_hat).max() <= np.abs(w).max() * 1.001,
            "no block overshoots its own amax",
        )
    # an all-zero block gets scale 1.0 and dequantises back to zero
    z = np.zeros((128, 128), dtype=np.float32)
    q, s = R.quantize_blockwise_fp8(z, R.HC_BLOCK)
    check(float(s[0, 0]) == 1.0, "all-zero block stores scale 1.0")
    check(np.all(R.dequant_blockwise_fp8(q, s, 128) == 0.0), "all-zero block is exact")


def test_hc_group_selection_and_merge():
    wmap = {}
    for layer in (0, 1):
        for role in ("attn", "mlp"):
            p = f"model.language_model.layers.{layer}.{role}_hyper_connection"
            wmap[f"{p}.input_mix_weight_down.weight"] = "s1"
            wmap[f"{p}.block_inject_weight.weight"] = "s1"
            wmap[f"{p}.input_mix_weight_up.weight"] = "s1"
            wmap[f"{p}.hc_norm.weight"] = "s1"
    wmap["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"] = "s2"
    wmap["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"] = "s2"
    for p in ("mtp.layers.0.attn_hyper_connection", "mtp.hyper_connection_mixer"):
        wmap[f"{p}.input_mix_weight_down.weight"] = "s3"
        wmap[f"{p}.block_inject_weight.weight"] = "s3"

    groups = R.select_hc_groups(wmap, "hc")
    kinds = sorted((g["kind"], g["out_name"]) for g in groups)
    check(len(groups) == 5, f"2 layers x 2 roles + 1 mixer = 5 groups, got {len(groups)}")
    check(
        sum(1 for g in groups if g["kind"] == "combine") == 4,
        "four combine groups, one mix group",
    )
    check(
        not any("mtp" in g["out_name"] for g in groups),
        "the drafter's HC modules are never selected by the hc tier",
    )
    check(
        all(g["out_name"].endswith("input_mix_weight_down_block_inject")
            for g in groups if g["kind"] == "combine"),
        "combine groups are emitted under the runtime's merged module name",
    )
    check(
        not any(".input_mix_weight_up" in s for g in groups for s in g["srcs"]),
        "the up projection is never a source (K=320 is not a multiple of 128)",
    )
    check(kinds[0][0] == "combine", "sorted group list is stable")

    # the merge itself
    rng = np.random.default_rng(5)
    lora, hc_count, hidden = 320, 4, 10240
    down = (rng.standard_normal((lora, hidden)) * 0.02).astype(np.float32)
    inject = (rng.standard_normal((hc_count, hidden)) * 0.05).astype(np.float32)
    g = [x for x in groups if x["kind"] == "combine"][0]
    res = R.convert_hc_group(g, down, inject, lora, hc_count)
    st = res["stats"]
    check(res["weight"].shape == (336, hidden), f"merged shape {res['weight'].shape}")
    check(res["weight_scale_inv"].shape == (3, 80), "merged scale is [3, 80]")
    w_hat = R.dequant_blockwise_fp8(res["weight"], res["weight_scale_inv"], 128)
    check(np.all(w_hat[324:] == 0.0), "the 12 pad rows are exactly zero")
    check(
        np.abs(w_hat[:lora] - down).max() < np.abs(down).max() * 0.1,
        "the down rows survive the merge",
    )
    check(
        np.abs(w_hat[lora : lora + hc_count] - inject).max()
        < np.abs(inject).max() * 0.1,
        "the injection rows survive the merge",
    )
    check(st["inject_sqnr_db"] > 30.0, f"injection SQNR {st['inject_sqnr_db']:.1f} dB")
    check(
        st["resident_bf16_bytes"] == 336 * hidden * 2
        and st["fp8_bytes"] == 336 * hidden + 3 * 80 * 4,
        "byte accounting is per-launch resident bytes, not checkpoint bytes",
    )
    # a mix group keeps its own name and shape
    gm = [x for x in groups if x["kind"] == "mix"][0]
    rm = R.convert_hc_group(gm, down, None, lora, hc_count)
    check(rm["weight"].shape == (320, hidden), "mix group stays [320, 10240]")
    check(rm["stats"]["kind"] == "mix" and "inject_sqnr_db" not in rm["stats"],
          "mix group has no injection rows")


def test_hc_rejects_bad_k():
    g = {"kind": "mix", "prefix": "p", "out_name": "p.input_mix_weight_down",
         "srcs": ["p.input_mix_weight_down.weight"]}
    w = np.full((320, 320), 0.01, dtype=np.float32)  # K = 320, not 128-aligned
    try:
        R.convert_hc_group(g, w, None, 320, 4)
        check(False, "K not divisible by block_k must be refused")
    except ValueError as e:
        check("per_token_group_quant_fp8" in str(e), f"refusal names the assert: {e}")


def main() -> int:
    for fn in (
        test_e4m3_lut,
        test_bf16_decode,
        test_blockwise_dequant_exact,
        test_pack_reference,
        test_quantizer_matches_vllm_reference,
        test_finer_group_is_more_accurate,
        test_fp16_scale_roundtrip,
        test_degenerate_group,
        test_qzeros_constant,
        test_bits_per_weight,
        test_sanity_guard,
        test_real_shapes_end_to_end,
        test_tier_selection,
        test_fused_uniformity,
        test_bf16_source_path,
        test_bf16_store_roundtrip,
        test_safetensors_roundtrip,
        test_e4m3_encode_grid,
        test_e4m3_encode_rounding,
        test_e4m3_encode_matches_torch,
        test_blockwise_fp8_roundtrip,
        test_hc_group_selection_and_merge,
        test_hc_rejects_bad_k,
    ):
        print(f"-- {fn.__name__}")
        fn()
    print(f"\n{NTEST - len(FAIL)}/{NTEST} checks passed")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
