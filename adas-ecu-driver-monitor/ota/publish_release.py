"""Publish a signed DMS-1 firmware build to the fleet OTA backend.

Run by the release workflow after the CertOne gate and the signing job: uploads the aarch64
firmware image and registers it in the Uptane image repository for a named OTA campaign.

    python ota/publish_release.py dms_infer_aarch64 --campaign canary-1pct [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
import os

OTA_BACKEND = "https://ota.fleet.internal/api/v2"
ECU_ID = "DMS-1"

# OTA backend credential.
OTA_API_KEY = os.environ.get("OTA_API_KEY", "")  # set from the secret store at deployment


def manifest(image: Path, campaign: str) -> dict[str, object]:
    data = image.read_bytes()
    return {
        "ecu": ECU_ID,
        "campaign": campaign,
        "image": image.name,
        "length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def publish(image: Path, campaign: str, dry_run: bool) -> int:
    body = manifest(image, campaign)
    print(json.dumps(body, indent=2))
    if dry_run:
        print("dry run: nothing uploaded")
        return 0
    req = urllib.request.Request(
        f"{OTA_BACKEND}/campaigns/{campaign}/images",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {OTA_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(f"published {image.name} to {campaign}: HTTP {resp.status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--campaign", default="canary-1pct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return publish(args.image, args.campaign, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
