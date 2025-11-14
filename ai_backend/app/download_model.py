#!/usr/bin/env python3
"""
download_model.py

Generic HTTP file downloader with simple resume support.
Usage:
  python download_model.py --url "https://example.com/model.gguf" --outname "vicuna-7b.gguf"

This script downloads the file into ./models/<outname>.
It supports:
 - Resuming a partial download (via Range header, if server supports it)
 - Optional SHA256 checksum verification via --sha256
 - Safe writing to a temporary file then rename on success
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib import request, error

CHUNK = 1024 * 1024  # 1MB


def download_url(url: str, out_path: Path, resume: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")

    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {}
    if resume and existing > 0:
        headers['Range'] = f'bytes={existing}-'

    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req) as resp:
            # If server returned 206 (Partial Content), we'll append to tmp file, else overwrite
            mode = "ab" if (resume and existing > 0 and resp.status == 206) else "wb"
            total = resp.getheader("Content-Length")
            if total is not None:
                try:
                    total = int(total) + (existing if mode == "ab" else 0)
                except Exception:
                    total = None
            downloaded = existing if mode == "ab" else 0

            print(f"Downloading to {out_path} (temp: {tmp_path})")
            if total:
                print(f"Total size: {total / (1024*1024):.2f} MB")

            with open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r{downloaded / (1024*1024):.2f} MB / {total / (1024*1024):.2f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\rDownloaded {downloaded / (1024*1024):.2f} MB", end="", flush=True)
            print("\nDownload complete, finalizing...")
            tmp_path.replace(out_path)
            return out_path
    except error.HTTPError as e:
        print("HTTP error:", e)
        raise
    except Exception as e:
        print("Download failed:", e)
        raise


def sha256_of_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download file to ./models/ and optionally verify sha256")
    parser.add_argument("--url", required=True, help="Direct URL to model file (http/https)")
    parser.add_argument("--outname", required=True, help="Filename to save under ./models/ (e.g. vicuna-7b.gguf)")
    parser.add_argument("--sha256", required=False, help="Optional SHA256 hex to verify after download")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume; always restart download")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / args.outname

    try:
        download_url(args.url, out_path, resume=not args.no_resume)
    except Exception as e:
        print("Error during download:", e)
        sys.exit(2)

    if args.sha256:
        print("Verifying SHA256...")
        got = sha256_of_file(out_path)
        if got.lower() != args.sha256.strip().lower():
            print("SHA256 mismatch!")
            print("Expected:", args.sha256)
            print("Got     :", got)
            sys.exit(3)
        print("SHA256 OK")

    print("Saved model to:", out_path)


if __name__ == "__main__":
    main()
