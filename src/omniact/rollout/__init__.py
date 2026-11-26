"""Remote closed-loop rollout — a `RolloutSession`-shaped gRPC protocol.

omniact-molmospaces serves the sim + FK (the env) and omniact drives the loop with a local policy, so the
two repos stay decoupled (no shared Python/torch/deps). See `session.py` for the contract and
`proto/rollout.proto` for the wire."""
from .policy import PolicyAdapter
from .record import write_rollout_wods
from .remote import RemoteRolloutSession
from .session import RolloutSession, last_tick


def read_bundle(path) -> dict[str, bytes]:
    """Read a replay bundle dir into ``{relative_path: bytes}``. ``path`` is any fsspec target -- a local dir
    or a ``gs://`` prefix -- so the client can source the bundle from anywhere and send its content, and the
    server needs no shared filesystem."""
    import fsspec

    fs, root = fsspec.core.url_to_fs(str(path))
    root = root.rstrip("/")
    files = fs.find(root)                       # recursive file list
    if not files:
        raise FileNotFoundError(f"no files under bundle path {path!r}")
    return {p[len(root):].lstrip("/"): fs.cat_file(p) for p in files}


def resolve_bundle(replay_root, wods_key, wods_root) -> dict[str, bytes]:
    """Resolve a WODS trajectory key to its complete replay bundle, HOUSE-AWARE, with no guessing.

    One embodiment (``gen_<seed>``) is reused across work-items, so ``<replay_root>/<bucket>/gen_<seed>/`` holds
    bundles from SEVERAL houses -- matching ``episode_idx`` alone is ambiguous (every house under the gen has
    ``traj00``/``traj01``) and silently picks the wrong scene. We match each bundle's per-episode-correct
    ``meta`` on the RELIABLE identifiers ``config_selector`` + house NUMBER + ``episode_idx``. We deliberately
    do NOT match ``scene_dataset``: the WODS key's scene can be stale (batch-last metadata bug), while
    ``(config, house_number, episode_idx)`` is verified UNAMBIGUOUS across the dataset (no gen has two bundles
    sharing all three across different scenes). RAISES (never guesses) if zero or more than one match. Assumes
    the ``replay/<seed//1e6:06d>/gen_<seed>/`` bucketing. ``replay_root`` is any fsspec target.

    The WODS ``caption`` (the authoritative per-episode instruction) is a LOGICAL part of the bundle -- the
    bundle's own ``meta['instruction']`` is batch-last-stale -- so we fold the WODS metadata (from ``wods_root``)
    into the bundle as ``wods_meta.json`` and return the complete ``{relative_path: bytes}`` map, letting the
    server read the caption from the self-contained bundle with no side-channel."""
    import json

    import fsspec

    gen = wods_key.split("/")[0]
    seed = int(gen.split("_")[1])
    config = wods_key.split("/")[1]
    house_num = wods_key.rsplit("_house_", 1)[1].split("_traj")[0]   # rsplit: robust to a scene containing '_house_'
    idx = int(wods_key.rsplit("_traj", 1)[1])
    fs, root = fsspec.core.url_to_fs(str(replay_root))
    gendir = f"{root.rstrip('/')}/{seed // 1_000_000:06d}/{gen}"
    try:
        seed_dirs = sorted(fs.ls(gendir, detail=False))
    except FileNotFoundError:
        seed_dirs = []
    matches = []
    for sd in seed_dirs:
        try:
            meta = json.loads(fs.cat_file(f"{sd}/meta.json"))
        except Exception:
            continue                                  # not a bundle dir / unreadable meta -> skip
        rc = meta.get("recon") or {}
        b_house = str(meta.get("house_id") or rc.get("house_id") or "").rsplit("_", 1)[-1]
        if rc.get("config_selector") == config and b_house == house_num and meta.get("episode_idx") == idx:
            matches.append(sd)
    if len(matches) != 1:
        raise FileNotFoundError(f"resolve_bundle found {len(matches)} bundles under {gendir} matching {wods_key!r} "
                                f"(config={config}, house={house_num}, episode_idx={idx}); expected exactly 1")
    rr = str(replay_root)                              # fs.ls drops the protocol; re-add it for remote roots
    path = f"{rr.split('://', 1)[0]}://{matches[0]}" if "://" in rr else matches[0]
    from wods.v2.item import read_wods_metadata
    bundle = read_bundle(path)
    bundle["wods_meta.json"] = json.dumps(read_wods_metadata(wods_root, wods_key)).encode()
    return bundle


def connect(target: str, *, bundle, options=None, channel=None) -> RemoteRolloutSession:
    """Open a remote rollout session on ``target`` (``host:port``). ``bundle`` is either a
    ``{relative_path: bytes}`` map or a path (local dir or ``gs://``) to read the replay bundle from; its
    content is sent to the server. ``options``: ``seed``, ``cameras`` (server-defined)."""
    if not isinstance(bundle, dict):
        bundle = read_bundle(bundle)
    return RemoteRolloutSession(target, bundle=bundle, options=dict(options or {}), channel=channel)


__all__ = ["RolloutSession", "RemoteRolloutSession", "PolicyAdapter", "connect", "last_tick", "read_bundle",
           "resolve_bundle", "write_rollout_wods"]
