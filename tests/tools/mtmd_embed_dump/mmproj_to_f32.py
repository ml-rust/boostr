#!/usr/bin/env python3
"""Rewrite an mmproj GGUF with every tensor dequantized to F32.

The encoder then runs f32 matmuls with the same weights, so a dump from it
isolates activation-quantization noise from the port's own error.

Usage: python3 mmproj_to_f32.py <in.gguf> <out.gguf>
Needs the `gguf` package from the reference checkout (`gguf-py`).
"""
import sys, numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize
src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = r.fields['general.architecture'].contents()
w = GGUFWriter(dst, arch)
for k, f in r.fields.items():
    if k.startswith('GGUF.') or k == 'general.architecture': continue
    v = f.contents()
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        sub = f.types[1]
        if sub == GGUFValueType.STRING: w.add_array(k, v)
        elif sub == GGUFValueType.BOOL: w.add_array(k, [bool(x) for x in v])
        elif sub == GGUFValueType.FLOAT32: w.add_array(k, [float(x) for x in v])
        else: w.add_array(k, [int(x) for x in v])
    elif t == GGUFValueType.STRING: w.add_string(k, v)
    elif t == GGUFValueType.BOOL: w.add_bool(k, bool(v))
    elif t == GGUFValueType.FLOAT32: w.add_float32(k, float(v))
    elif t == GGUFValueType.UINT32: w.add_uint32(k, int(v))
    elif t == GGUFValueType.INT32: w.add_int32(k, int(v))
    else: raise SystemExit(f'unhandled {k} {t}')
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape)]  # row-major
    if t.tensor_type == GGMLQuantizationType.F32:
        a = np.array(t.data, dtype=np.float32).reshape(shape)
    elif t.tensor_type == GGMLQuantizationType.F16:
        a = np.array(t.data, dtype=np.float16).astype(np.float32).reshape(shape)
    else:
        a = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape)
    w.add_tensor(t.name, a, raw_dtype=GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('wrote', dst)
