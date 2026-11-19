# -*- coding: utf-8 -*-
"""Find houses and panoramas in a folder that contains only panorama images.

Layout: <root>/<house>/**/<image>.jpg  (each sub-folder of root is one house).
If root itself holds images and no sub-folders with images, root is a single house.
Only 2:1 images are kept. Images are identified by file stem; identical content found
twice keeps the first copy, different content with the same stem gets a numbered suffix.
"""
import hashlib
import os

from PIL import Image

from . import config


def _images_under(folder):
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in config.EXCLUDE_DIRS)
        for f in sorted(filenames):
            if f.lower().endswith(config.IMAGE_EXTS):
                out.append(os.path.join(dirpath, f))
    return out


def _sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n))           # first MiB is enough to tell copies apart
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


def scan_houses(root):
    """Return {house: [{"id", "path", "w", "h"}]} (absolute paths) and skipped files with reasons."""
    root = os.path.abspath(root)
    subs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))
            and d.lower() not in config.EXCLUDE_DIRS]
    candidates = {d: _images_under(os.path.join(root, d)) for d in subs}
    candidates = {d: v for d, v in candidates.items() if v}
    if not candidates:
        direct = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(config.IMAGE_EXTS)]
        candidates = {os.path.basename(root): direct}
    houses, skipped = {}, []
    for house, paths in candidates.items():
        seen_hash, seen_id, items = {}, set(), []
        for p in paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:                     # unreadable file
                skipped.append(dict(path=p, reason="unreadable: %s" % e)); continue
            if h == 0 or abs(w / h - 2.0) > 0.02:
                skipped.append(dict(path=p, reason="not 2:1 (%dx%d)" % (w, h))); continue
            digest = _sha(p)
            if digest in seen_hash:
                skipped.append(dict(path=p, reason="duplicate of " + seen_hash[digest])); continue
            stem = os.path.splitext(os.path.basename(p))[0]
            pid, k = stem, 2
            while pid in seen_id:
                pid = "%s_%d" % (stem, k); k += 1
            seen_id.add(pid); seen_hash[digest] = p
            items.append(dict(id=pid, path=p, w=w, h=h))
        if len(items) >= 2:
            houses[house] = items
        else:
            skipped.append(dict(path=house, reason="house has fewer than 2 panoramas"))
    return houses, skipped


def all_pairs(items):
    """Every unordered pair, deterministic order."""
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]
