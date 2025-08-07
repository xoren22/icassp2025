import os
import time
import numpy as np
from helper import load_samples
from approx import (
    load_precomputed_normals_for_building,
    calculate_combined_loss_with_normals,
)


def main():
    s = load_samples(num_samples=1, seed=1)[0]
    refl, trans, _ = s.input_img.cpu().numpy()
    b_id = s.ids[0] if s.ids else 0

    print(f"Sample: B{b_id}, shape={refl.shape[::-1]} (W×H)")

    t0 = time.perf_counter()
    nx, ny = load_precomputed_normals_for_building(b_id, refl, trans)
    t1 = time.perf_counter()
    print(f"load_normals: {(t1-t0):.4f}s, nx.dtype={nx.dtype}, C={nx.flags['C_CONTIGUOUS']}")

    # Ensure contiguous inputs
    refl_c = np.ascontiguousarray(refl, dtype=np.float64)
    trans_c = np.ascontiguousarray(trans, dtype=np.float64)

    # Warmup
    _ = calculate_combined_loss_with_normals(
        refl_c, trans_c, nx, ny,
        s.x_ant, s.y_ant, s.freq_MHz,
        max_refl=0, max_trans=1,
        n_angles=64,
        radial_step=1.0,
    )

    # Measure two full runs
    for i in range(2):
        t0 = time.perf_counter()
        _ = calculate_combined_loss_with_normals(
            refl_c, trans_c, nx, ny,
            s.x_ant, s.y_ant, s.freq_MHz,
            max_refl=5, max_trans=15,
            n_angles=360*128,
            radial_step=1.0,
        )
        t1 = time.perf_counter()
        print(f"kernel_run[{i}]: {(t1-t0):.3f}s")


if __name__ == "__main__":
    main() 