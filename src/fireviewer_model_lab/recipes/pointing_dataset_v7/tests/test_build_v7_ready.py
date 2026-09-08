from training.pointing_dataset_v7.build_v7_ready import assign_groupwise_splits, cap_recurrence, enforce_distribution_caps


def sample(sha: str, group: str, source: str = "source", category: int | None = 1) -> dict:
    categories = [] if category is None else [category]
    return {
        "sha256": sha,
        "split_group_id": group,
        "source_dataset": source,
        "objects": {"category": categories, "bbox": [] if category is None else [[0, 0, 1, 1]]},
    }


def test_groupwise_assignment_is_deterministic_and_non_leaking() -> None:
    rows = [sample(f"{index:064x}", f"group-{index // 2}") for index in range(40)]
    first = assign_groupwise_splits(rows)
    second = assign_groupwise_splits(list(reversed(rows)))
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert len(first) == 20


def test_recurrence_cap_is_group_bounded_and_deterministic() -> None:
    rows = [sample(f"{index:064x}", "one-group") for index in range(20)]
    kept, audit = cap_recurrence(rows, maximum=12)
    repeated, repeated_audit = cap_recurrence(list(reversed(rows)), maximum=12)
    assert len(kept) == 12
    assert audit == {"maximum_per_group": 12, "removed": 8, "group_count": 1}
    assert [row["sha256"] for row in kept] == [row["sha256"] for row in repeated]
    assert audit == repeated_audit


def test_distribution_caps_bound_source_and_negatives() -> None:
    rows = [sample(f"{index:064x}", f"group-{index}", "dominant", 0) | {"scene_bin": "fire_medium", "v7_origin": "v6_reviewed_base"} for index in range(70)]
    rows += [sample(f"{100 + index:064x}", f"other-{index}", "other", 1) | {"scene_bin": "smoke_only", "v7_origin": "v6_reviewed_base"} for index in range(20)]
    rows += [sample(f"{200 + index:064x}", f"negative-{index}", "other", None) | {"scene_bin": "hard_negative", "v7_origin": "v4_reviewed_negative_reuse"} for index in range(10)]
    kept, audit = enforce_distribution_caps(rows, source_cap=0.70, negative_cap=0.10)
    assert max(sum(row["source_dataset"] == source for row in kept) / len(kept) for source in {"dominant", "other"}) <= 0.70
    assert sum(not row["objects"]["bbox"] for row in kept) / len(kept) <= 0.10
    assert audit["source_cap"] == 0.70
