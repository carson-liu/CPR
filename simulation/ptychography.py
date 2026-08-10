"""Forward simulation for the AD_LTEM-derived ptychography dataset.

The numerical operations and defaults follow the original dataset notebook.
"""

from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftshift
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.transform import resize
from tqdm import tqdm


def load_and_preprocess_image(image_path, new_size):
    image = imread(image_path).astype(np.float32)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
    return resize(image, new_size)


def create_complex_object(amplitude, phase):
    return amplitude * np.exp(1j * phase)


def adjust_amplitude_phase(amplitude, phase):
    amplitude_adjusted = (
        0.5
        * (amplitude - amplitude.min())
        / (amplitude.max() - amplitude.min())
        + 0.5
    )
    phase_adjusted = -np.pi / 3 * (phase + np.pi) / (2 * np.pi)
    return amplitude_adjusted, phase_adjusted


def simulate_diffraction_with_amp_phase(
    probe,
    amplitude,
    phase,
    pos,
    add_noise=False,
    noise_type="gaussian",
    noise_level=0.01,
    apply_jitter=False,
    jitter=0,
):
    """Simulate one far-field intensity pattern and its object patch."""
    if apply_jitter:
        jitter_x, jitter_y = np.random.randint(-jitter, jitter + 1, 2)
        pos_jittered = (
            max(min(pos[0] + jitter_y, amplitude.shape[0] - probe.shape[0]), 0),
            max(min(pos[1] + jitter_x, amplitude.shape[1] - probe.shape[1]), 0),
        )
    else:
        pos_jittered = pos

    y, x = pos_jittered
    amp_section = amplitude[y : y + probe.shape[0], x : x + probe.shape[1]]
    phase_section = phase[y : y + probe.shape[0], x : x + probe.shape[1]]
    obj_section = amp_section * np.exp(1j * phase_section)

    if obj_section.shape != probe.shape:
        raise ValueError(
            f"Object section shape {obj_section.shape} does not match "
            f"probe shape {probe.shape}."
        )

    diffraction = np.abs(fftshift(fft2(obj_section * probe))) ** 2

    if add_noise:
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, diffraction.shape)
            diffraction += noise
        if noise_type == "gaussian and poisson":
            noise = np.random.normal(0, noise_level, diffraction.shape)
            diffraction += noise
            diffraction = np.clip(diffraction, 0, None)
            diffraction = np.random.poisson(diffraction)
        elif noise_type == "poisson":
            diffraction = np.random.poisson(diffraction)

    return diffraction, amp_section, phase_section


def calculate_step_size(probe_size, overlap_rate):
    return int(probe_size * (1 - overlap_rate / 100))


def load_and_scale_probe(probe_path, block_size=(4, 4)):
    probe = np.load(probe_path).astype(np.complex64)
    scale_func = lambda part: block_reduce(  # noqa: E731
        part, block_size=block_size, func=np.mean
    )
    scaled_real = scale_func(probe.real)
    scaled_imag = scale_func(probe.imag)
    return probe, scaled_real + 1j * scaled_imag


def load_and_reshape(data_list, target_shape):
    data = np.array(data_list)
    assert int(np.sqrt(data.shape[0])) ** 2 == data.shape[0], (
        "Data does not form a perfect square."
    )
    return data.reshape(target_shape)


def normalize_data(data, target_range=(0, 1)):
    scale = (target_range[1] - target_range[0]) / (data.max() - data.min())
    return (data - data.min()) * scale + target_range[0]


def resize_images(
    data,
    new_size=(32, 32),
    mode="reflect",
    anti_aliasing=True,
    apply_resize=True,
):
    if not apply_resize:
        return data
    resized_data = np.zeros(
        (data.shape[0], data.shape[1]) + new_size, dtype=data.dtype
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            resized_data[i, j] = resize(
                data[i, j], new_size, mode=mode, anti_aliasing=anti_aliasing
            )
    return resized_data


def resize_center_region(
    data,
    new_size=(32, 32),
    preserve_range=True,
    anti_aliasing=True,
    apply_resize=True,
):
    if not apply_resize:
        return data
    original_cut_size = new_size[0] // 2
    center_region_resized = np.zeros(
        (data.shape[0], data.shape[1]) + new_size, dtype=data.dtype
    )
    for i in tqdm(range(data.shape[0]), desc="Resizing diffraction patterns"):
        for j in range(data.shape[1]):
            center_slice = data[
                i,
                j,
                data.shape[2] // 2
                - original_cut_size : data.shape[2] // 2
                + original_cut_size,
                data.shape[3] // 2
                - original_cut_size : data.shape[3] // 2
                + original_cut_size,
            ]
            center_region_resized[i, j] = resize(
                center_slice,
                new_size,
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing,
            )
    return center_region_resized


def _print_data_summary(prefix, amplitude, phase, diffraction):
    print(
        f"{prefix} shapes - Amplitude: {amplitude.shape}, "
        f"Phase: {phase.shape}, Diffraction: {diffraction.shape}"
    )
    print(f"{prefix} amplitude range: min={amplitude.min()}, max={amplitude.max()}")
    print(f"{prefix} phase range: min={phase.min()}, max={phase.max()}")
    print(
        f"{prefix} diffraction range: min={diffraction.min()}, "
        f"max={diffraction.max()}"
    )


def generate_ad_ltem_dataset(
    amplitude_path,
    phase_path,
    probe_path,
    output_dir=".",
    image_size=(512, 512),
    overlap_rate=75,
    probe_block_size=(4, 4),
    output_patch_size=(32, 32),
    add_noise=True,
    noise_type="gaussian and poisson",
    noise_level=0.01,
    apply_jitter=True,
    jitter=0,
    seed=None,
):
    """Generate and save the raw and normalized AD_LTEM arrays.

    A seed is optional because the original notebook used NumPy's unseeded
    global random number generator. Passing a seed makes the noisy diffraction
    realization repeatable without changing the simulation operations.
    """
    if seed is not None:
        np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amplitude = load_and_preprocess_image(amplitude_path, image_size)
    phase = load_and_preprocess_image(phase_path, image_size)
    _, probe = load_and_scale_probe(probe_path, block_size=probe_block_size)

    amplitude, phase = adjust_amplitude_phase(amplitude, phase)
    object_complex = create_complex_object(amplitude, phase)

    step_size = calculate_step_size(probe.shape[0], overlap_rate)
    if step_size <= 0:
        raise ValueError("The overlap rate must produce a positive scan step.")
    positions = [
        (y, x)
        for y in range(0, object_complex.shape[0] - probe.shape[0] + 1, step_size)
        for x in range(0, object_complex.shape[1] - probe.shape[1] + 1, step_size)
    ]

    simulated = [
        simulate_diffraction_with_amp_phase(
            probe,
            amplitude,
            phase,
            pos,
            add_noise=add_noise,
            noise_type=noise_type,
            noise_level=noise_level,
            apply_jitter=apply_jitter,
            jitter=jitter,
        )
        for pos in tqdm(positions, desc="Simulating scan positions")
    ]
    diffraction_patterns, amplitude_sections, phase_sections = zip(*simulated)

    dimension_side = int(np.sqrt(len(phase_sections)))
    patch_shape = np.asarray(phase_sections).shape[1:]
    new_shape = (dimension_side, dimension_side) + patch_shape
    amplitude_raw = load_and_reshape(amplitude_sections, new_shape)
    phase_raw = load_and_reshape(phase_sections, new_shape)
    diffraction_raw = load_and_reshape(diffraction_patterns, new_shape)
    _print_data_summary("Raw data", amplitude_raw, phase_raw, diffraction_raw)

    file_prefix = f"{overlap_rate:g}"
    np.save(output_dir / f"{file_prefix}_amp.npy", amplitude_raw)
    np.save(output_dir / f"{file_prefix}_ph.npy", phase_raw)
    np.save(output_dir / f"{file_prefix}_diff.npy", diffraction_raw)

    amplitude_normalized = resize_images(
        normalize_data(amplitude_raw), new_size=output_patch_size, apply_resize=True
    )
    phase_normalized = resize_images(
        normalize_data(phase_raw), new_size=output_patch_size, apply_resize=True
    )
    diffraction_normalized = resize_center_region(
        normalize_data(diffraction_raw),
        new_size=output_patch_size,
        apply_resize=True,
    )
    _print_data_summary(
        "Normalized data",
        amplitude_normalized,
        phase_normalized,
        diffraction_normalized,
    )

    np.save(output_dir / f"{file_prefix}_amp_n.npy", amplitude_normalized)
    np.save(output_dir / f"{file_prefix}_ph_n.npy", phase_normalized)
    np.save(output_dir / f"{file_prefix}_diff_n.npy", diffraction_normalized)

    return {
        "amplitude": amplitude_raw,
        "phase": phase_raw,
        "diffraction": diffraction_raw,
        "amplitude_normalized": amplitude_normalized,
        "phase_normalized": phase_normalized,
        "diffraction_normalized": diffraction_normalized,
    }
