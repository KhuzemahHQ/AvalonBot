"""Talking-head clips for the seats.

Images live in the repo `avatars` folder and are assigned to seats 0–4 in
alphabetical filename order. This module never logs a filename or the pixels.
One file type the video API rejects (AVIF) is re-encoded in memory first.
"""

import hashlib
import io
from pathlib import Path
from typing import Optional, Tuple

import aiohttp

from server.audio_proxy import BOSON_API_KEY

FACE_DIR = Path(__file__).resolve().parents[1] / "avatars"
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".avif"}
_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".avif": "image/avif",
}
STREAM_URL = "https://api.boson.ai/v1/videos/stream"

_faces: Optional[list] = None
_face_sig: Optional[tuple] = None


def faces() -> list:
    """Seat order is alphabetical by filename. The names stay in the folder.

    Replacing or adding a file is picked up on the next request. The log is only a count.
    """
    global _faces, _face_sig
    found = []
    if FACE_DIR.is_dir():
        found = sorted(
            (p for p in FACE_DIR.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT),
            key=lambda p: p.name.lower(),
        )
    sig = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in found)
    if sig != _face_sig:
        _face_sig = sig
        _faces = found
        print(f"[faces] {len(found)} images assigned to seats")
    return _faces or []


def face_stamp() -> str:
    """Changes when a portrait file is replaced. Contains no filenames."""
    digest = hashlib.sha256()
    for mtime, size in _face_sig or ():
        digest.update(f"{mtime}:{size}".encode())
    return digest.hexdigest()[:12]


def face_count() -> int:
    return len(faces())


def still_for_seat(seat: int) -> Optional[Tuple[bytes, str]]:
    """Original file bytes and a content type, for the portrait between lines."""
    paths = faces()
    if seat < 0 or seat >= len(paths):
        return None
    path = paths[seat]
    return path.read_bytes(), _MIME[path.suffix.lower()]


def _upload_image(seat: int) -> Optional[Tuple[bytes, str, str]]:
    """Bytes the video API will accept. AVIF is rewritten to JPEG in memory."""
    paths = faces()
    if seat < 0 or seat >= len(paths):
        return None
    path = paths[seat]
    suffix = path.suffix.lower()
    if suffix == ".avif":
        from PIL import Image

        image = Image.open(path)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg", "face.jpg"
    return path.read_bytes(), _MIME[suffix], "face" + suffix


async def iter_clip_chunks(seat: int, wav: bytes):
    """Yields fragmented-MP4 bytes as Higgs generates them. Playback can start on the first frames."""
    if not BOSON_API_KEY or not wav:
        return
    prepared = _upload_image(seat)
    if prepared is None:
        return
    image, mime, filename = prepared
    timeout = aiohttp.ClientTimeout(total=180, sock_read=180)
    headers = {"Authorization": f"Bearer {BOSON_API_KEY}"}
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(2):
                form = aiohttp.FormData()
                form.add_field("model", "higgs-avatar")
                form.add_field("size", "480x640")
                form.add_field("ref_image", image, filename=filename, content_type=mime)
                form.add_field("input", wav, filename="line.wav", content_type="audio/wav")
                async with session.post(STREAM_URL, data=form, headers=headers) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    if resp.status == 429 and attempt == 0:
                        await resp.read()
                        await _pause()
                        continue
                    if resp.status != 200 or "json" in ctype:
                        err = (await resp.text())[:300]
                        print(f"[faces] seat {seat} clip failed {resp.status}: {err}")
                        return
                    async for chunk in resp.content.iter_chunked(8192):
                        if chunk:
                            yield chunk
                    return
            print(f"[faces] seat {seat} clip failed 429")
    except Exception as exc:
        print(f"[faces] seat {seat} clip error: {type(exc).__name__}")


async def _pause() -> None:
    import asyncio
    await asyncio.sleep(2.0)
