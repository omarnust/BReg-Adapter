from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_xml(xml_path: Path) -> tuple[str, str]:
    """
    Return (filename, wnid) from an ILSVRC annotation XML.

    NOTE: Uses the first <object> only (ImageNet single-label convention).
    """
    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")
    if not filename:
        raise ValueError("missing <filename>")

    obj = root.find("object")
    if obj is None:
        raise ValueError("missing <object>")

    wnid = obj.findtext("name")
    if not wnid:
        raise ValueError("missing <object><name>")

    return filename, wnid


def _resolve_image_path(images_dir: Path, filename: str) -> Path:
    """
    Resolve an image path from XML filename.
    ILSVRC XMLs sometimes omit extensions; images are typically *.JPEG.
    """
    candidates = []

    p = images_dir / filename
    candidates.append(p)
    if p.exists():
        return p

    if Path(filename).suffix == "":
        p2 = images_dir / f"{filename}.JPEG"
        candidates.append(p2)
        if p2.exists():
            return p2

    p3 = images_dir / f"{Path(filename).stem}.JPEG"
    candidates.append(p3)
    if p3.exists():
        return p3

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve image for '{filename}'. Tried: {tried}")


def _safe_rel_symlink(target: Path, link_path: Path) -> None:
    """Create a symlink using a relative target when possible."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = os.path.relpath(target, start=link_path.parent)
        link_path.symlink_to(rel)
    except Exception:
        link_path.symlink_to(target)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Create ImageFolder val/<wnid>/... view from ILSVRC val XMLs"
    )
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--ann-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--link-type",
        choices=["symlink", "hardlink"],
        default="symlink",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-errors", type=int, default=50)

    args = ap.parse_args(argv)

    if not args.images_dir.exists():
        print(f"ERROR: images-dir does not exist: {args.images_dir}", file=sys.stderr)
        return 2
    if not args.ann_dir.exists():
        print(f"ERROR: ann-dir does not exist: {args.ann_dir}", file=sys.stderr)
        return 2

    xmls = sorted(args.ann_dir.glob("*.xml"))
    if not xmls:
        print(f"ERROR: no XML files found in: {args.ann_dir}", file=sys.stderr)
        return 2

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        print(
            f"ERROR: out-dir exists and is not empty: {args.out_dir}",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    would_create = 0
    skipped = 0
    errors = 0

    for i, xml_path in enumerate(xmls, start=1):
        try:
            filename, wnid = _parse_xml(xml_path)
            img_path = _resolve_image_path(args.images_dir, filename)

            dst_dir = args.out_dir / wnid
            dst_path = dst_dir / img_path.name

            # IMPORTANT: use os.path.lexists to catch broken symlinks
            if os.path.lexists(dst_path):
                skipped += 1
                continue

            if args.dry_run:
                print(f"[{i}/{len(xmls)}] {args.link_type}: {dst_path} -> {img_path}")
                would_create += 1
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)

            if args.link_type == "symlink":
                _safe_rel_symlink(img_path, dst_path)
            else:
                try:
                    os.link(img_path, dst_path)
                except OSError as e:
                    raise RuntimeError(
                        "Hardlink failed (possible cross-filesystem link)"
                    ) from e

            created += 1

        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"WARN: {xml_path.name}: {e}", file=sys.stderr)
            if errors >= args.max_errors:
                print("ERROR: too many errors, aborting.", file=sys.stderr)
                break

    if args.dry_run:
        print(
            f"Done (dry-run). would_create={would_create}, skipped={skipped}, errors={errors}"
        )
    else:
        print(
            f"Done. created={created}, skipped={skipped}, errors={errors}, out-dir={args.out_dir}"
        )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
