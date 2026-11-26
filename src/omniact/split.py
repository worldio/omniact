from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


SPLIT_ALGORITHM = "sha256-axis-v1"
VAL_FRACTION_DENOMINATOR = 20
VAL_UNSEEN_EMBODIMENT = "unseen_embodiment"
VAL_UNSEEN_SCENE = "unseen_scene"
VAL_UNSEEN_BOTH = "unseen_embodiment_and_scene"
VALIDATION_SPLITS = (VAL_UNSEEN_EMBODIMENT, VAL_UNSEEN_SCENE, VAL_UNSEEN_BOTH)


@dataclass(frozen=True)
class OmniActKeySplit:
    train_keys: frozenset[str]
    validation_keys: dict[str, frozenset[str]]


def key_set_sha256(keys: Iterable[str]) -> str:
    """Hash a key set canonically, independent of its input order."""
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _split_score(identifier: str, *, axis: str, seed: int) -> bytes:
    payload = f"{SPLIT_ALGORITHM}\0{seed}\0{axis}\0{identifier}".encode()
    return hashlib.sha256(payload).digest()


def select_training_embodiments(
    embodiments: Iterable[str],
    count: int,
    seed: int,
) -> frozenset[str]:
    """Select an exact, nested, seeded prefix of eligible training embodiments."""
    available = set(embodiments)
    if count < 0:
        raise ValueError(f"subsample_train_embodiments must be >= 0, got {count}")
    if count == 0:
        return frozenset(available)
    if count > len(available):
        raise ValueError(
            f"subsample_train_embodiments={count} exceeds "
            f"{len(available)} eligible training embodiments"
        )
    ordered = sorted(
        available,
        key=lambda value: (_split_score(value, axis="training_subset", seed=seed), value),
    )
    return frozenset(ordered[:count])


def select_training_trajectories(
    keys: Iterable[str],
    fraction: float,
    seed: int,
) -> frozenset[str]:
    """Thin each embodiment's trajectories to a seeded, nested fraction.

    Holds the embodiment set fixed and subsamples within each one, so the corpus
    shrinks without losing diversity -- the complement of
    :func:`select_training_embodiments`. With the same seed, smaller fractions are
    subsets of larger ones. Every embodiment keeps at least one trajectory, so the
    embodiment count is preserved. A fraction of 1.0 retains everything.
    """
    selected = set(keys)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_trajectory_fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return frozenset(selected)
    by_embodiment: dict[str, list[str]] = defaultdict(list)
    for key in selected:
        by_embodiment[_split_key(key)[1]].append(key)
    kept: set[str] = set()
    for embodiment_keys in by_embodiment.values():
        ordered = sorted(
            embodiment_keys,
            key=lambda value: (_split_score(value, axis="training_trajectory", seed=seed), value),
        )
        kept.update(ordered[: max(1, round(len(ordered) * fraction))])
    return frozenset(kept)


def _validation_ids(identifiers: Iterable[str], *, axis: str, seed: int) -> frozenset[str]:
    unique = set(identifiers)
    if len(unique) < 2:
        return frozenset()
    ordered = sorted(unique, key=lambda value: (_split_score(value, axis=axis, seed=seed), value))
    count = min(max(1, len(ordered) // VAL_FRACTION_DENOMINATOR), len(ordered) - 1)
    return frozenset(ordered[:count])


def _split_key(key: str) -> tuple[str, str, str]:
    """Parse a molmospaces WODS key into ``(scene, embodiment, task)``.

    Keys are ``gen_<seed>/<task>/<scene>_house_<h>_traj<NN>``. The embodiment is ``gen_<seed>``, the
    task is the config name, and the scene axis is the key's ``<scene>_house_<h>`` label, so
    trajectories sharing that label are held out together."""
    try:
        embodiment, task, trajectory = key.split("/", 2)
        scene, _index = trajectory.rsplit("_traj", 1)
    except ValueError as error:
        raise ValueError(
            f"invalid OmniAct key {key!r}; expected gen_<seed>/<task>/<scene>_traj<NN>"
        ) from error
    if not embodiment or not task or not scene:
        raise ValueError(
            f"invalid OmniAct key {key!r}; expected gen_<seed>/<task>/<scene>_traj<NN>"
        )
    return scene, embodiment, task


def split_train_val_keys(
    keys: Iterable[str],
    seed: int,
    tasks: Iterable[str] | None = None,
    *,
    subsample_train_embodiments: int = 0,
    train_trajectory_fraction: float = 1.0,
) -> OmniActKeySplit:
    """Reserve axes globally, then classify validation by final training exposure.

    Optional embodiment subsampling follows task filtering; discarded training
    embodiments become unseen in validation. Zero retains all eligible embodiments.
    Trajectory thinning follows it and keeps every surviving embodiment, so the two
    knobs shrink the corpus along independent axes.
    """
    ordered_keys = sorted(keys)
    if not ordered_keys:
        raise ValueError("cannot split an empty key universe")
    if any(left == right for left, right in zip(ordered_keys, ordered_keys[1:])):
        raise ValueError("OmniAct key universe contains duplicates")

    embodiments = set()
    scenes_by_task: dict[str, set[str]] = defaultdict(set)
    for key in ordered_keys:
        scene, embodiment, task = _split_key(key)
        embodiments.add(embodiment)
        scenes_by_task[task].add(scene)

    task_prefixes = tuple(tasks or ())
    held_out_embodiments = _validation_ids(embodiments, axis="embodiment", seed=seed)
    held_out_scenes = frozenset(
        scene
        for task, scenes in scenes_by_task.items()
        for scene in _validation_ids(scenes, axis=f"scene:{task}", seed=seed)
    )

    train_keys = set()
    for key in ordered_keys:
        scene, embodiment, task = _split_key(key)
        if task_prefixes and not task.startswith(task_prefixes):
            continue
        if embodiment not in held_out_embodiments and scene not in held_out_scenes:
            train_keys.add(key)
    if not train_keys:
        raise ValueError("seeded scene/embodiment split produced an empty training set")

    train_embodiments = select_training_embodiments(
        (_split_key(key)[1] for key in train_keys),
        subsample_train_embodiments,
        seed,
    )
    train_keys = {key for key in train_keys if _split_key(key)[1] in train_embodiments}
    train_keys = set(select_training_trajectories(train_keys, train_trajectory_fraction, seed))

    # Sparse coverage and embodiment subsampling can remove all training pairs for
    # a scene. Classify validation against the exposure that actually remains.
    train_scenes = frozenset(_split_key(key)[0] for key in train_keys)

    val_unseen_embodiment_keys = set()
    val_unseen_scene_keys = set()
    val_unseen_both_keys = set()
    for key in ordered_keys:
        if key in train_keys:
            continue
        scene, embodiment, task = _split_key(key)
        if task_prefixes and not task.startswith(task_prefixes):
            continue
        unseen_embodiment = embodiment not in train_embodiments
        unseen_scene = scene not in train_scenes
        if unseen_embodiment and unseen_scene:
            val_unseen_both_keys.add(key)
        elif unseen_embodiment:
            val_unseen_embodiment_keys.add(key)
        elif unseen_scene:
            val_unseen_scene_keys.add(key)
        else:
            raise AssertionError(f"non-training key {key!r} is seen on both split axes")

    return OmniActKeySplit(
        train_keys=frozenset(train_keys),
        validation_keys={
            VAL_UNSEEN_EMBODIMENT: frozenset(val_unseen_embodiment_keys),
            VAL_UNSEEN_SCENE: frozenset(val_unseen_scene_keys),
            VAL_UNSEEN_BOTH: frozenset(val_unseen_both_keys),
        },
    )
