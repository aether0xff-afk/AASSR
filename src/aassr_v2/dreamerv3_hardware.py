from __future__ import annotations

from typing import Any, MutableMapping


CANONICAL_DREAMERV3_ACCELERATOR = "gpu"


def actual_jax_hardware_manifest(requested_platform: str) -> dict[str, Any]:
    """Inspect the JAX devices after DreamerV3 has initialized its backend.

    `requested_platform` is configuration intent; this function records the
    devices JAX actually created so a canonical artifact cannot claim CUDA while
    silently executing on CPU.
    """

    try:
        import jax
    except ImportError as exc:
        raise RuntimeError("JAX is unavailable while verifying DreamerV3 hardware") from exc

    devices = tuple(jax.devices())
    if not devices:
        raise RuntimeError("JAX reported no devices")
    platforms = tuple(str(device.platform) for device in devices)
    device_strings = tuple(str(device) for device in devices)
    requested = str(requested_platform).lower()
    accelerator_required = requested in {"cuda", "gpu"}
    accelerator_present = CANONICAL_DREAMERV3_ACCELERATOR in platforms
    if accelerator_required and not accelerator_present:
        raise RuntimeError(
            "DreamerV3 requested CUDA/GPU but JAX created no GPU device: "
            f"{device_strings}"
        )
    return {
        "requested_platform": requested,
        "actual_platforms": list(platforms),
        "devices": list(device_strings),
        "device_count": len(devices),
        "accelerator_required": accelerator_required,
        "accelerator_present": accelerator_present,
    }


def stamp_dreamer_summary_hardware(
    summary: MutableMapping[str, Any],
    *,
    requested_platform: str,
) -> dict[str, Any]:
    hardware = actual_jax_hardware_manifest(requested_platform)
    summary["jax_hardware"] = hardware
    return hardware
