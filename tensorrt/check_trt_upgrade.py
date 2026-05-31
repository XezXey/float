"""
check_trt_upgrade.py — TensorRT & CUDA Environment Inspector
=============================================================
Inspects the current environment and reports:
  - Installed TensorRT version (all cu11/cu12 variants)
  - Available upgrade versions from PyPI
  - CUDA toolkit / driver compatibility
  - Whether known-buggy TRT versions for MHA fusions are in use
  - Recommended upgrade command

Usage:
  python tensorrt/check_trt_upgrade.py
  python tensorrt/check_trt_upgrade.py --apply          # attempt the upgrade
  python tensorrt/check_trt_upgrade.py --target 10.9.0  # pin to a specific version
"""

from __future__ import annotations

import os
import sys
import json
import re
import subprocess
import argparse
import textwrap
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Known-buggy TRT versions for MHA-related regressions
# References:
#   NVIDIA/TensorRT#4796 — attention layer-fusion mismatch
#   NVIDIA/TensorRT#3609 — sequence slice regression in multi-head attention
# ─────────────────────────────────────────────────────────────────────────────
_KNOWN_BUGGY_VERSIONS = {
    "10.0.1": ["#4796 adaLN Slice/Gather INT64 chain corruption", "#3609 MHA sequence-slice regression"],
    "10.1.0": ["#4796 partial fix — adaLN chain still affected"],
    "10.2.0": ["#4796 partial fix — adaLN chain still affected"],
    "10.3.0": ["#4796 adaLN Slice/Gather INT64 chain corruption", "#3609 MHA sequence-slice regression"],
    "10.4.0": ["#3609 MHA sequence-slice — partially fixed"],
    "10.5.0": ["#3609 MHA sequence-slice — partially fixed"],
    "10.6.0": ["#3609 MHA — improved, residual risk on sm_89 with cuDNN Flash Attn"],
}

# First version known to fully fix both issues:
_RECOMMENDED_MIN_VERSION = "10.7.0"
_LATEST_STABLE = "10.16.1"    # as of 2026-05; update as needed


def _run_pip_show(*packages) -> dict[str, dict]:
    """Return pip show output for each package as a dict keyed by package name."""
    results = {}
    for pkg in packages:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                meta = {}
                for line in r.stdout.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                results[pkg] = meta
        except Exception:
            pass
    return results


def _get_available_trt_versions() -> list[str]:
    """Query PyPI for available tensorrt versions."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "tensorrt"],
            capture_output=True, text=True, timeout=30,
        )
        # Example: "Available versions: 11.0.0.114, 10.16.1.11, ..."
        m = re.search(r"Available versions:\s*(.+)", r.stdout)
        if m:
            raw = [v.strip() for v in m.group(1).split(",")]
            # Normalise "10.16.1.11" → "10.16.1" (strip build suffix)
            versions = []
            for v in raw:
                parts = v.split(".")
                # TRT versions are major.minor.patch[.build]
                if len(parts) >= 3:
                    versions.append(".".join(parts[:3]))
            return versions
    except Exception:
        pass
    return []


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except ValueError:
        return (0,)


def _detect_active_trt() -> Optional[str]:
    """Return the version string of the TRT that `import tensorrt` resolves to."""
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return None


def _detect_cuda() -> dict:
    info: dict = {}
    # nvcc
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        m = re.search(r"release (\S+),", r.stdout)
        if m:
            info["nvcc"] = m.group(1).rstrip(",")
    except FileNotFoundError:
        info["nvcc"] = "not found"

    # nvidia-smi
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                             "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            for i, line in enumerate(r.stdout.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                info[f"gpu_{i}"] = {
                    "name": parts[0] if len(parts) > 0 else "?",
                    "driver": parts[1] if len(parts) > 1 else "?",
                    "memory_mib": parts[2] if len(parts) > 2 else "?",
                }
    except FileNotFoundError:
        info["nvidia_smi"] = "not found"

    # PyTorch CUDA
    try:
        import torch
        info["pytorch_cuda"] = torch.version.cuda
        info["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            p = torch.cuda.get_device_properties(0)
            info["compute_capability"] = f"{p.major}.{p.minor}"
            info["vram_gb"] = round(p.total_memory / 1e9, 1)
    except Exception:
        pass

    return info


def _upgrade_command(target_version: Optional[str], cu: str = "cu12") -> str:
    """Return the recommended pip install command."""
    if target_version:
        return (
            f"pip install tensorrt=={target_version} "
            f"tensorrt-{cu}=={target_version}"
        )
    return (
        f"pip install --upgrade tensorrt tensorrt-{cu}"
    )


def _ort_gpu_install_command() -> str:
    return "pip install onnxruntime-gpu  # also uninstall onnxruntime first"


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────
def print_report(args):
    print("=" * 64)
    print("TensorRT & CUDA Environment Inspector")
    print("=" * 64)

    # ── CUDA info ──────────────────────────────────────────────────────────────
    cuda = _detect_cuda()
    print("\n  ── CUDA / GPU ──────────────────────────────────────────────")
    for k, v in cuda.items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                print(f"      {kk:20s}: {vv}")
        else:
            print(f"    {k:24s}: {v}")

    # ── TRT versions ───────────────────────────────────────────────────────────
    print("\n  ── TensorRT Installed Packages ─────────────────────────────")
    pkgs = _run_pip_show("tensorrt", "tensorrt-cu11", "tensorrt-cu12",
                         "tensorrt_cu11", "tensorrt_cu12")
    if not pkgs:
        print("    [none found]")
    for pkg, meta in pkgs.items():
        ver = meta.get("Version", "?")
        loc = meta.get("Location", "?")
        req = meta.get("Required-by", "?")
        print(f"    {pkg:20s}: v{ver}   (required-by: {req})")
        print(f"    {'':20s}  @ {loc}")

    # ── active TRT ────────────────────────────────────────────────────────────
    print("\n  ── Active `import tensorrt` ────────────────────────────────")
    active = _detect_active_trt()
    if active is None:
        print("    [FAIL] Cannot import tensorrt. Check PYTHONPATH and LD_LIBRARY_PATH.")
    else:
        print(f"    Version: {active}")
        bugs = _known_bugs(active)
        if bugs:
            print(f"    ❌  Known bugs in {active}:")
            for b in bugs:
                print(f"       • {b}")
        else:
            print(f"    ✅  No known MHA regression bugs in {active}.")

    # ── available upgrades ─────────────────────────────────────────────────────
    print("\n  ── Available TensorRT Upgrades (PyPI) ─────────────────────")
    available = _get_available_trt_versions()
    if not available:
        print("    [WARN] Could not reach PyPI. Check network connectivity.")
    else:
        # Show versions newer than active
        active_t = _version_tuple(active or "0.0.0")
        newer = [v for v in available if _version_tuple(v) > active_t][:10]
        recommended_min = _version_tuple(_RECOMMENDED_MIN_VERSION)

        if not newer:
            print(f"    Already at or above latest PyPI release ({available[0]}).")
        else:
            print(f"    Newer versions available (newest first):")
            for v in newer[:8]:
                tag = ""
                if _version_tuple(v) >= _version_tuple(_LATEST_STABLE):
                    tag = "  ← latest stable"
                elif _version_tuple(v) >= recommended_min:
                    tag = "  ← MHA bug fixes included"
                elif v in _KNOWN_BUGGY_VERSIONS:
                    tag = "  ← known MHA bug"
                print(f"      {v}{tag}")

    # ── recommendation ──────────────────────────────────────────────────────
    print("\n  ── Recommendation ──────────────────────────────────────────")
    if active is not None:
        active_t = _version_tuple(active)
        rec_t    = _version_tuple(_RECOMMENDED_MIN_VERSION)
        if active_t < rec_t:
            print(f"    ⚠️  {active} is older than recommended minimum ({_RECOMMENDED_MIN_VERSION}).")
            print(f"    Upgrade command:")
            print(f"      {_upgrade_command(args.target or _RECOMMENDED_MIN_VERSION)}")
            print()
            print(f"    Full upgrade to latest stable ({_LATEST_STABLE}):")
            print(f"      {_upgrade_command(None)}")
        else:
            print(f"    ✅  {active} meets the minimum version requirement ({_RECOMMENDED_MIN_VERSION}).")
            print(f"    If bugs persist, try upgrading to latest:")
            print(f"      {_upgrade_command(None)}")

    # ── onnxruntime-gpu check ─────────────────────────────────────────────────
    print("\n  ── ONNX Runtime Provider ───────────────────────────────────")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        print(f"    onnxruntime version : {ort.__version__}")
        print(f"    Providers           : {providers}")
        if has_cuda:
            print("    ✅  CUDA EP available — ORT sanity checks will use GPU.")
        else:
            print("    ⚠️  CUDA EP NOT available — ORT sanity checks run on CPU only.")
            print("    To enable GPU ORT:")
            print(f"      pip uninstall onnxruntime -y && {_ort_gpu_install_command()}")
    except ImportError:
        print("    [WARN] onnxruntime not installed. Install with:")
        print("      pip install onnxruntime-gpu")

    # ── LD_LIBRARY_PATH hint ───────────────────────────────────────────────────
    print("\n  ── Library Path Notes ──────────────────────────────────────")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld:
        print("    LD_LIBRARY_PATH is unset. If TRT shared libs fail to load:")
        print("      export LD_LIBRARY_PATH=/usr/local/lib/python3.x/dist-packages/tensorrt_libs:$LD_LIBRARY_PATH")
    else:
        trt_in_ld = any("tensorrt" in p.lower() for p in ld.split(":"))
        if trt_in_ld:
            print(f"    ✅  TRT paths found in LD_LIBRARY_PATH.")
        else:
            print(f"    ⚠️  LD_LIBRARY_PATH set but no tensorrt paths found.")
            print(f"    Current value: {ld[:200]}")

    print()


def _known_bugs(version: str) -> list[str]:
    """Return list of known bug strings for a given version string."""
    # Strip build suffix: "10.3.0" from "10.3.0.post1" or "10.3.0.114"
    base = ".".join(version.split(".")[:3])
    return _KNOWN_BUGGY_VERSIONS.get(base, [])


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade logic
# ─────────────────────────────────────────────────────────────────────────────
def do_upgrade(args):
    active = _detect_active_trt()
    cuda = _detect_cuda()

    # Determine cu suffix from PyTorch CUDA version
    pt_cuda = cuda.get("pytorch_cuda", "12.x")
    cu = "cu11" if pt_cuda and pt_cuda.startswith("11") else "cu12"
    print(f"  CUDA toolkit: {pt_cuda} → using {cu} TRT wheel family")

    # Determine target version
    target = args.target
    if not target:
        available = _get_available_trt_versions()
        # Filter to cu-compatible, find recommended minimum or newer
        candidates = [
            v for v in available
            if _version_tuple(v) >= _version_tuple(_RECOMMENDED_MIN_VERSION)
        ]
        if not candidates:
            print("  [WARN] No candidates found from PyPI. Check network.")
            return False
        target = candidates[0]  # PyPI returns newest first

    print(f"  Target version: {target}")
    active_t = _version_tuple(active or "0.0.0")
    target_t = _version_tuple(target)
    if active_t >= target_t:
        print(f"  Active TRT ({active}) is already ≥ target ({target}). Skipping.")
        return True

    cmd = [sys.executable, "-m", "pip", "install",
           f"tensorrt=={target}", f"tensorrt-{cu}=={target}"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"\n  ✅  TensorRT upgraded to {target}.")
        print("  ⚠️  You MUST rebuild your .engine files — engines are NOT cross-version compatible.")
        print("  Run:")
        print("    ./tensorrt/run_pipeline.sh --precision fp16 --skip_onnx")
        return True
    else:
        print(f"\n  ❌  Upgrade failed (exit code {result.returncode}).")
        print("  Try manually:")
        print(f"    {_upgrade_command(target, cu)}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect and optionally upgrade TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Just inspect (no changes):
          python tensorrt/check_trt_upgrade.py

          # Inspect and auto-upgrade to recommended minimum:
          python tensorrt/check_trt_upgrade.py --apply

          # Pin to a specific version:
          python tensorrt/check_trt_upgrade.py --apply --target 10.9.0
        """),
    )
    p.add_argument("--apply",  action="store_true",
                   help="Apply the upgrade (pip install). Default: report only.")
    p.add_argument("--target", type=str, default=None,
                   help=f"Pin to a specific TRT version (default: {_RECOMMENDED_MIN_VERSION}+).")
    p.add_argument("--json_out", type=str, default=None,
                   help="Write JSON report to this path.")
    return p.parse_args()


def main():
    args = parse_args()
    print_report(args)

    if args.apply:
        print("=" * 64)
        print("Applying TensorRT Upgrade")
        print("=" * 64)
        do_upgrade(args)
    else:
        print("  To apply the recommended upgrade, run:")
        print("    python tensorrt/check_trt_upgrade.py --apply")
        print()


if __name__ == "__main__":
    main()
