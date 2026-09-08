from __future__ import annotations

from fireviewer_model_lab.training.wikimedia_corpus import canonical_open_license, resume_candidate_titles


def test_canonical_open_license_accepts_only_supported_creative_commons_urls() -> None:
    assert canonical_open_license(
        "CC BY-SA 4.0", "http://creativecommons.org/licenses/by-sa/4.0/"
    ) == (
        "CC-BY-SA-4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
    )
    assert canonical_open_license("CC0", "https://creativecommons.org/publicdomain/zero/1.0/") == (
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    )


def test_canonical_open_license_maps_structured_public_domain_labels() -> None:
    assert canonical_open_license("CC0", "") == (
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    )
    assert canonical_open_license("Public domain", "") == (
        "PDM-1.0",
        "https://creativecommons.org/publicdomain/mark/1.0/",
    )


def test_canonical_open_license_rejects_untrusted_or_restricted_licenses() -> None:
    assert canonical_open_license("CC BY 4.0", "") is None
    assert canonical_open_license("Public domain", "https://example.com/license") is None
    assert (
        canonical_open_license("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/")
        is None
    )


def test_resume_includes_new_and_transient_titles_only() -> None:
    assert resume_candidate_titles(
        ["File:accepted.jpg", "File:permanent.gif", "File:transient.jpg", "File:new.jpg"],
        accepted_titles={"File:accepted.jpg"},
        permanent_rejection_titles={"File:permanent.gif"},
    ) == ["File:new.jpg", "File:transient.jpg"]
