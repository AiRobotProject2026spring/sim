#!/usr/bin/env python3
"""Download a pi0 checkpoint to shared storage.

Run this from a compute/interactive job, not from the login node, unless your
cluster policy explicitly allows large network downloads on login nodes.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


DEFAULT_DEST = Path("/work/gw13/share/pi0")
DEFAULT_REPO_ID = "lerobot/pi0"


def looks_like_login_node() -> bool:
    hostname = socket.gethostname().lower()
    login_markers = ("login", "head", "front", "gateway")
    return any(marker in hostname for marker in login_markers)


def download_from_huggingface(
    repo_id: str,
    revision: str | None,
    dest: Path,
    token: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  pixi add --pypi huggingface_hub\n"
            "or run this script from an environment that already provides it."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=dest,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )


def download_from_gcs(gcs_uri: str, dest: Path) -> None:
    if not gcs_uri.startswith("gs://"):
        raise SystemExit(f"GCS source must start with gs://, got: {gcs_uri}")

    cmd = ["gsutil", "-m", "rsync", "-r", gcs_uri, str(dest)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: gsutil. Load your site's Google Cloud SDK "
            "module or install gsutil before using --source gcs."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"gsutil failed with exit code {exc.returncode}") from exc


def parse_csv_patterns(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pi0 model files into shared storage."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--source",
        choices=("hf", "gcs"),
        default="hf",
        help="Download backend. Use hf for Hugging Face, gcs for gs:// paths.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--gcs-uri",
        default=None,
        help="GCS checkpoint URI, for example gs://bucket/path.",
    )
    parser.add_argument(
        "--allow-patterns",
        default=None,
        help="Comma-separated Hugging Face allow patterns, e.g. '*.safetensors,*.json'.",
    )
    parser.add_argument(
        "--ignore-patterns",
        default=None,
        help="Comma-separated Hugging Face ignore patterns.",
    )
    parser.add_argument(
        "--allow-login-node",
        action="store_true",
        help="Bypass the login-node safety check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if looks_like_login_node() and not args.allow_login_node:
        print(
            "Refusing to start a large download on what looks like a login node "
            f"({socket.gethostname()}).\n"
            "Submit an interactive/batch job and run this script there, or pass "
            "--allow-login-node if your cluster policy permits it.",
            file=sys.stderr,
        )
        return 2

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Destination: {dest}")
    if args.source == "hf":
        print(f"Downloading Hugging Face model: {args.repo_id}")
        download_from_huggingface(
            repo_id=args.repo_id,
            revision=args.revision,
            dest=dest,
            token=args.token,
            allow_patterns=parse_csv_patterns(args.allow_patterns),
            ignore_patterns=parse_csv_patterns(args.ignore_patterns),
        )
    else:
        if not args.gcs_uri:
            raise SystemExit("--gcs-uri is required when --source gcs")
        print(f"Downloading GCS checkpoint: {args.gcs_uri}")
        download_from_gcs(args.gcs_uri, dest)

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
