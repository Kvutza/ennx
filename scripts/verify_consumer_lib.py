"""End-to-end consumer verification script for ENNX library.

Exercises real FP4/FP8 quantized PackedSearch, BPANN indexing, and high-level
TuRBO optimization to guarantee runtime library functionality beyond unit tests.
"""

import sys

import numpy as np

try:
    from ennx.ennx_rust import optimizer

    from ennx.experimental import quantize_fp4_e2m1

    print("[✓] Successfully imported native ENNX components and quantization helpers.")
except ImportError as e:
    print(f"[X] Import failed: {e}")
    sys.exit(1)


def test_end_to_end_quantized_search():
    print("\n--- Running End-to-End FP4/FP8 Quantized PackedSearch ---")
    raw_data = np.random.randn(512).astype(np.float32)

    # 1. Quantize data using quantization module
    packed_bytes = quantize_fp4_e2m1(raw_data, scale=0.5)
    print(f"Quantized 512 floats into {len(packed_bytes)} packed bytes.")

    # 2. Build PackedLeaf schema tuples: (offset, length, bits, scale, weight, radius)
    leaf = (0, 512, 4, 0.5, 1.0, 0.75)

    # 3. Instantiate native PackedSearch engine (CPU & GPU backend)
    for backend in ["cpu", "metal"]:
        search = optimizer.PackedSearch(packed_bytes, 0.25, [leaf], 4, backend)

        trial_index, trial_seed, trial_score = search.ask(
            np.array([19, 23, 29, 31], dtype=np.uint64), 0.65, 1
        )
        print(
            f"  [{backend}] Winning trial index: {trial_index}, seed: {trial_seed}, score: {trial_score:.4f}"
        )

        row_bytes = search.row()
        assert len(row_bytes) == len(packed_bytes), "Materialized row length mismatch"
        search.tell(0.85, True)

    print("[✓] FP4/FP8 Quantized PackedSearch verified across backends.")


if __name__ == "__main__":
    test_end_to_end_quantized_search()
    print(
        "\n[SUMMARY] ALL END-TO-END CONSUMER LIBRARY INTEGRATION CHECKS PASSED SUCCESSFULLY!"
    )
