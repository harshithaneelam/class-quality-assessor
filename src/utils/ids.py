"""Resolve public short IDs (e.g. "good_1") to the local video_index keys."""
from __future__ import annotations


def resolve_short_id(index: dict, short_id: str) -> str:
    """Return the index key equal to `short_id` or starting with `short_id + "_"`.

    Local index keys may carry extra suffixes that must not appear in the public repo.
    """
    if short_id in index:
        return short_id
    hits = [k for k in index if k.startswith(short_id + "_")]
    if len(hits) != 1:
        raise KeyError(f"short id {short_id!r} matched {len(hits)} index entries")
    return hits[0]
