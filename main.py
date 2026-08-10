"""Train, evaluate, or generate data for CPR."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    load_ad_ltem_test_set,
    load_experimental_test_set,
    prepare_ad_ltem_data,
    prepare_experimental_data,
)
from simulation import generate_ad_ltem_dataset
from utils import (
    AdaptiveImageStitcher,
    ExperimentalImageStitcher,
    average_patch_metrics,
    build_model_from_checkpoint,
    format_average_metrics,
    format_metrics,
    load_checkpoint,
    plot_global_stitching,
    plot_patch_comparison,
    reconstruct_patches,
    reconstruction_metrics,
    select_device,
    train_model,
)


REPO_ROOT = Path(__file__).resolve().parent
SIMULATION_ASSETS = REPO_ROOT / "simulation" / "assets"

DATASETS = {
    "ad_ltem": {
        "prepare": prepare_ad_ltem_data,
        "load_test": load_ad_ltem_test_set,
        "grid_size": 12,
        "validation_size": 100,
        "patch_indices": (0, 33, 65, 110, 143),
        "stitcher": lambda: AdaptiveImageStitcher(
            point_size=8, overlap=5 * 8
        ),
    },
    "experimental": {
        "prepare": prepare_experimental_data,
        "load_test": load_experimental_test_set,
        "grid_size": 60,
        "validation_size": 805,
        "patch_indices": (0, 915, 1830, 2745, 3599),
        "stitcher": lambda: ExperimentalImageStitcher(
            point_size=3, patch_size=64, output_size=(60, 60)
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="ad_ltem",
        help="Dataset to use (default: ad_ltem).",
    )
    parser.add_argument(
        "--mode",
        choices=("test", "train", "generate"),
        default="test",
        help="Operation to run (default: test).",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)

    generation = parser.add_argument_group("AD_LTEM data generation")
    generation.add_argument(
        "--amplitude",
        type=Path,
        default=SIMULATION_ASSETS / "amplitude.tiff",
    )
    generation.add_argument(
        "--phase", type=Path, default=SIMULATION_ASSETS / "phase.tiff"
    )
    generation.add_argument(
        "--probe", type=Path, default=SIMULATION_ASSETS / "probe.npy"
    )
    generation.add_argument("--image-size", type=int, default=512)
    generation.add_argument("--overlap-rate", type=float, default=75)
    generation.add_argument("--probe-block-size", type=int, default=4)
    generation.add_argument("--patch-size", type=int, default=32)
    generation.add_argument("--noise-level", type=float, default=0.01)
    generation.add_argument(
        "--noise-type",
        choices=("gaussian", "poisson", "gaussian and poisson"),
        default="gaussian and poisson",
    )
    generation.add_argument("--jitter", type=int, default=1)
    generation.add_argument("--seed", type=int, default=None)
    generation.add_argument("--no-noise", action="store_true")
    return parser.parse_args()


def checkpoint_path(args):
    if args.checkpoint is not None:
        return args.checkpoint
    return REPO_ROOT / "checkpoints" / args.dataset / "best_model.pth"


def run_train(args):
    settings = DATASETS[args.dataset]
    validation_size = args.validation_size
    if validation_size is None:
        validation_size = settings["validation_size"]
    prepared_data = settings["prepare"](
        args.data_root,
        grid_size=settings["grid_size"],
        validation_size=validation_size,
    )
    train_model(
        prepared_data,
        dataset_name=args.dataset,
        checkpoint_path=checkpoint_path(args),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
    )


def run_test(args):
    settings = DATASETS[args.dataset]
    output_dir = args.output_dir or REPO_ROOT / "results" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or select_device()

    checkpoint = load_checkpoint(checkpoint_path(args))
    if checkpoint.get("dataset") not in (None, args.dataset):
        raise ValueError(
            f"Checkpoint dataset is {checkpoint.get('dataset')!r}, "
            f"not {args.dataset!r}."
        )
    test_data = settings["load_test"](
        args.data_root, grid_size=settings["grid_size"]
    )
    model = build_model_from_checkpoint(checkpoint, device=device)
    amp_pred, phase_pred = reconstruct_patches(
        model, test_data.diffraction, args.batch_size, device
    )

    amp_average = average_patch_metrics(test_data.amplitude, amp_pred)
    phase_average = average_patch_metrics(test_data.phase, phase_pred)
    print(f"All {len(amp_pred)} {args.dataset} patches (mean ± std)")
    print(f"Amplitude: {format_average_metrics(amp_average)}")
    print(f"Phase:     {format_average_metrics(phase_average)}")

    stitcher = settings["stitcher"]()
    global_amp_true = stitcher.stitch(
        test_data.amplitude, test_data.grid_size
    )
    global_amp_pred = stitcher.stitch(amp_pred, test_data.grid_size)
    global_phase_true = stitcher.stitch(
        test_data.phase, test_data.grid_size
    )
    global_phase_pred = stitcher.stitch(phase_pred, test_data.grid_size)
    global_amp_metrics = reconstruction_metrics(global_amp_true, global_amp_pred)
    global_phase_metrics = reconstruction_metrics(
        global_phase_true, global_phase_pred
    )
    print("Global stitched metrics")
    print(f"Amplitude: {format_metrics(global_amp_metrics)}")
    print(f"Phase:     {format_metrics(global_phase_metrics)}")

    np.save(output_dir / "amplitude.npy", amp_pred)
    np.save(output_dir / "phase.npy", phase_pred)
    patch_figure, _ = plot_patch_comparison(
        test_data.diffraction,
        test_data.amplitude,
        amp_pred,
        test_data.phase,
        phase_pred,
        indices=settings["patch_indices"],
        grid_size=test_data.grid_size,
        save_path=output_dir / "patch_comparison.png",
    )
    global_figure = plot_global_stitching(
        global_amp_true,
        global_amp_pred,
        global_phase_true,
        global_phase_pred,
        global_amp_metrics,
        global_phase_metrics,
        save_path=output_dir / "global_stitching.png",
    )
    plt.close(patch_figure)
    plt.close(global_figure)
    print(f"Results saved to {output_dir}")


def run_generate(args):
    if args.dataset != "ad_ltem":
        raise ValueError("Data generation is only available for AD_LTEM.")
    output_dir = args.output_dir or args.data_root
    generate_ad_ltem_dataset(
        amplitude_path=args.amplitude,
        phase_path=args.phase,
        probe_path=args.probe,
        output_dir=output_dir,
        image_size=(args.image_size, args.image_size),
        overlap_rate=args.overlap_rate,
        probe_block_size=(args.probe_block_size, args.probe_block_size),
        output_patch_size=(args.patch_size, args.patch_size),
        add_noise=not args.no_noise,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        apply_jitter=True,
        jitter=args.jitter,
        seed=args.seed,
    )


def main():
    args = parse_args()
    operations = {
        "train": run_train,
        "test": run_test,
        "generate": run_generate,
    }
    operations[args.mode](args)


if __name__ == "__main__":
    main()
