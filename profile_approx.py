import timeit

import numpy as np
from approx import (
    calculate_combined_loss,
    calculate_transmission_loss_numpy,
    MAX_REFL,
    MAX_TRANS,
)
from helper import load_samples


def profile_once(n_angles: int = 360 * 128, repeats: int = 3, seed: int | None = 42):
    """Profile both ray-tracing routines on a single random sample.

    Parameters
    ----------
    n_angles : int
        Number of rays shot per routine.  Defaults match the library.
    repeats : int
        How many repetitions to average over for each routine (after JIT warm-up).
    """

    # ---- prepare input sample ------------------------------------------------
    sample = load_samples(num_samples=1, seed=seed)[0]
    ref, trans, _ = sample.input_img.cpu().numpy()
    x_ant, y_ant, freq_MHz = sample.x_ant, sample.y_ant, sample.freq_MHz

    # ---- JIT warm-up to avoid compilation cost in timing ---------------------
    print("Warming up numba JIT …", flush=True)
    calculate_combined_loss(
        ref,
        trans,
        x_ant,
        y_ant,
        freq_MHz,
        n_angles=n_angles,
        max_refl=MAX_REFL,
        max_trans=MAX_TRANS,
    )
    calculate_transmission_loss_numpy(
        trans,
        x_ant,
        y_ant,
        freq_MHz,
        n_angles=n_angles,
        max_walls=MAX_TRANS,
    )

    # ---- define callables for timeit ----------------------------------------

    def _run_combined():
        calculate_combined_loss(
            ref,
            trans,
            x_ant,
            y_ant,
            freq_MHz,
            n_angles=n_angles,
            max_refl=MAX_REFL,
            max_trans=MAX_TRANS,
        )

    def _run_trans_only():
        calculate_transmission_loss_numpy(
            trans,
            x_ant,
            y_ant,
            freq_MHz,
            n_angles=n_angles,
            max_walls=MAX_TRANS,
        )

    # ---- timing -------------------------------------------------------------
    comb_time_total = timeit.timeit(_run_combined, number=repeats)
    trans_time_total = timeit.timeit(_run_trans_only, number=repeats)

    print("\n---- Profiling Results ----")
    print(f"Combined   (avg of {repeats}): {comb_time_total / repeats:.3f} s")
    print(f"Transmission-only (avg of {repeats}): {trans_time_total / repeats:.3f} s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile execution times of ray-tracing approximation routines."
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=360 * 128,
        help="Number of angular steps (default: 360*128)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions for averaging (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection",
    )

    args = parser.parse_args()

    profile_once(n_angles=args.angles, repeats=args.repeats, seed=args.seed) 