#!/usr/bin/env python3
"""
setup_projgan.py
~~~~~~~~~~~~~~~~
Clone the autonomousvision/projected-gan repository and apply
6 PyTorch 2.x / modern-timm compatibility patches (the same ones



"""

import argparse
import os
import re
import subprocess
import sys

REPO_URL = "https://github.com/autonomousvision/projected-gan.git"

# ── helpers ────────────────────────────────────────────────────────────

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _replace_in_file(filepath: str, old: str, new: str, label: str) -> bool:
    """Replace *old* with *new* in *filepath*.  Returns True if a change was made."""
    if not os.path.exists(filepath):
        print(f"  [{label}] File not found: {filepath}")
        return False
    code = _read(filepath)
    if old not in code:
        print(f"  [{label}] Already OK: {os.path.basename(filepath)}")
        return False
    code = code.replace(old, new)
    _write(filepath, code)
    print(f"  [{label}] Patched: {os.path.basename(filepath)}")
    return True


# ── individual patches ─────────────────────────────────────────────────

def patch1_infinite_sampler(pgan_dir: str) -> None:
    """PATCH 1: InfiniteSampler – remove deprecated ``dataset`` arg from
    ``super().__init__(dataset)`` in torch_utils/misc.py (and distributed.py
    if present)."""
    print("PATCH 1: InfiniteSampler super().__init__")
    for fn in ("torch_utils/misc.py", "torch_utils/distributed.py"):
        fp = os.path.join(pgan_dir, fn)
        _replace_in_file(fp, "super().__init__(dataset)", "super().__init__()", "Sampler")


def patch2_torch_amp(pgan_dir: str) -> None:
    """PATCH 2: torch.cuda.amp → torch.amp (PyTorch ≥2.x)."""
    print("PATCH 2: torch.cuda.amp -> torch.amp")
    for fn in ("training/training_loop.py", "training/loss.py"):
        fp = os.path.join(pgan_dir, fn)
        if not os.path.exists(fp):
            print(f"  [amp] File not found: {fp}")
            continue
        code = _read(fp)
        new_code = code
        new_code = new_code.replace(
            "torch.cuda.amp.autocast(enabled=",
            "torch.amp.autocast('cuda', enabled=",
        )
        new_code = new_code.replace(
            "torch.cuda.amp.GradScaler(enabled=",
            "torch.amp.GradScaler('cuda', enabled=",
        )
        if new_code != code:
            _write(fp, new_code)
            print(f"  [amp] Patched: {os.path.basename(fp)}")
        else:
            print(f"  [amp] Already OK: {os.path.basename(fp)}")


def patch3_copy_params_and_buffers(pgan_dir: str) -> None:
    """PATCH 3: copy_params_and_buffers – add ``persistent_only=True`` to
    ``named_buffers()`` calls in torch_utils/misc.py so that non-persistent
    buffers (added by newer PyTorch) are skipped."""
    print("PATCH 3: copy_params_and_buffers persistent_only")
    fp = os.path.join(pgan_dir, "torch_utils/misc.py")
    if not os.path.exists(fp):
        print(f"  [copy_params] File not found: {fp}")
        return
    code = _read(fp)
    # The original repo has bare .named_buffers() – we add persistent_only=True.
    old = ".named_buffers()"
    new = ".named_buffers(persistent_only=True)"
    if old in code and new not in code:
        code = code.replace(old, new)
        _write(fp, code)
        print("  [copy_params] Patched")
    else:
        print("  [copy_params] Already OK")


def patch4_timm_act_layer(pgan_dir: str) -> None:
    """PATCH 5 (notebook numbering): timm ≥0.9 renamed ``act1`` →
    ``act_layer`` on EfficientNet stems.  We replace the hard-coded
    ``model.act1`` reference with a ``getattr`` fallback so both old
    and new timm work."""
    print("PATCH 4: timm act_layer (act1 fallback)")
    # The notebook patches pg_modules/projector.py; the user mentions
    # pg_modules/blocks.py.  We patch whichever file(s) contain the pattern.
    for fn in ("pg_modules/projector.py", "pg_modules/blocks.py"):
        fp = os.path.join(pgan_dir, fn)
        if not os.path.exists(fp):
            continue
        code = _read(fp)
        # Only patch if the hard-coded reference exists and we haven't
        # already injected the getattr guard.
        if "model.act1" not in code or "hasattr" in code or "getattr" in code:
            print(f"  [timm/act1] Already OK: {os.path.basename(fp)}")
            continue
        # Two known patterns in the repo:
        # 1) projector.py: "model.conv_stem, model.bn1, model.act1,"
        old_pattern = "model.conv_stem, model.bn1, model.act1,"
        new_pattern = (
            "model.conv_stem, model.bn1, "
            'getattr(model, "act1", getattr(model, "act_conv", nn.SiLU(inplace=True))),'
        )
        if old_pattern in code:
            code = code.replace(old_pattern, new_pattern)
            _write(fp, code)
            print(f"  [timm/act1] Patched: {os.path.basename(fp)}")
        else:
            # 2) blocks.py: "self.model.conv_stem.act1"
            old2 = "self.model.conv_stem.act1"
            new2 = (
                'getattr(self.model.conv_stem, "act_layer", '
                'getattr(self.model.conv_stem, "act1", nn.SiLU(inplace=True)))'
            )
            if old2 in code:
                code = code.replace(old2, new2)
                _write(fp, code)
                print(f"  [timm/act1] Patched: {os.path.basename(fp)}")
            else:
                print(f"  [timm/act1] No matching pattern in {os.path.basename(fp)}")


def patch5_distributed_stub(pgan_dir: str) -> None:
    """PATCH 5: If ``torch_utils/distributed.py`` does not exist, create a
    minimal stub with an ``InfiniteSampler`` that does *not* inherit from
    ``torch.utils.data.Sampler`` (avoids the deprecated-super().__init__
    issue altogether)."""
    print("PATCH 5: distributed.py stub")
    fp = os.path.join(pgan_dir, "torch_utils/distributed.py")
    if os.path.exists(fp):
        print("  [distributed] File already exists – skipping stub creation")
        return

    stub = '''\
"""Minimal distributed utilities stub for projected-gan.

This file is auto-generated by setup_projgan.py to provide an
``InfiniteSampler`` that works without inheriting from
``torch.utils.data.Sampler``.
"""

import torch
import numpy as np


class InfiniteSampler:
    """Wraps a dataset index sampler that loops forever."""

    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = int(np.rint(order.size * self.window_size))
        epoch = 0
        while True:
            if self.shuffle:
                rnd = np.random.RandomState(seed=self.seed + epoch)
                rnd.shuffle(order)
                epoch += 1
            idx = 0
            while idx < order.size:
                i = idx % order.size
                if window >= 2:
                    j = (i - rnd.randint(window)) % order.size if rnd is not None else i
                else:
                    j = i
                if j % self.num_replicas == self.rank:
                    yield order[j]
                idx += 1
'''
    _write(fp, stub)
    print("  [distributed] Created stub")


def patch6_adam_betas(pgan_dir: str) -> None:
    """PATCH 6: Adam ``betas=(0, 0.99)`` → ``betas=(0.0, 0.99)`` so that
    PyTorch does not receive an int where it expects a float."""
    print("PATCH 6: Adam betas int -> float")
    for fn in ("train.py", "training/training_loop.py"):
        fp = os.path.join(pgan_dir, fn)
        if not os.path.exists(fp):
            print(f"  [betas] File not found: {fn}")
            continue
        code = _read(fp)
        # Use a regex that matches betas=[0, ...] or betas=(0, ...) with a
        # bare ``0`` (integer) as the first element.
        fixed = re.sub(r"""(betas['"]?\s*[:=]\s*[\[(])\s*0\s*,""", r"\g<1>0.0,", code)
        if fixed != code:
            _write(fp, fixed)
            print(f"  [betas] Patched: {fn}")
        else:
            print(f"  [betas] Already OK: {fn}")


# ── main ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone projected-gan and apply PyTorch 2.x compatibility patches."
    )
    # Default: <stylegan_root>/projected_gan
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "projected_gan",
    )
    parser.add_argument(
        "--projgan_dir",
        type=str,
        default=default_dir,
        help="Path to the projected-gan directory (default: %(default)s)",
    )
    args = parser.parse_args()
    pgan_dir = os.path.abspath(args.projgan_dir)

    # ── clone if needed ────────────────────────────────────────────────
    if not os.path.isdir(pgan_dir):
        print(f"Cloning projected-gan into {pgan_dir} ...")
        ret = subprocess.call(["git", "clone", REPO_URL, pgan_dir])
        if ret != 0:
            print("ERROR: git clone failed.", file=sys.stderr)
            return 1
        print("Clone complete.\n")
    else:
        print(f"ProjectedGAN repo already exists at {pgan_dir}\n")

    # ── apply patches ──────────────────────────────────────────────────
    print("Applying PyTorch 2.x compatibility patches ...\n")

    patch1_infinite_sampler(pgan_dir)
    print()
    patch2_torch_amp(pgan_dir)
    print()
    patch3_copy_params_and_buffers(pgan_dir)
    print()
    patch4_timm_act_layer(pgan_dir)
    print()
    patch5_distributed_stub(pgan_dir)
    print()
    patch6_adam_betas(pgan_dir)

    print("\nAll 6 patches applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
