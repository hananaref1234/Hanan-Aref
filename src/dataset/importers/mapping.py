"""Safe class-mapping rules. Never invent directional or doorway labels."""

from __future__ import annotations

from dataclasses import dataclass

from src.dataset.manifest import MappingSpec, SourceSpec
from src.dataset.taxonomy import COMBINED_CLASSES, CUSTOM_CLASSES, TAXONOMY_VERSION

MAPPING_TYPES = (
    "DIRECT",
    "MANUAL_DIRECTION",
    "SEMANTIC_REVIEW",
    "MASK_TO_BOX",
    "CUSTOM",
)

# Source names that must never be auto-assigned a directional Li-Stick class.
DIRECTIONAL_SOURCE_CLASSES = frozenset(
    {
        "stairs",
        "stair",
        "step",
        "steps",
        "curb",
        "kerb",
    }
)

DIRECTIONAL_TARGETS = frozenset(
    {
        "stairs_up",
        "stairs_down",
        "curb_up",
        "curb_down",
    }
)

# Explicit pairs that code must refuse even if a caller asks to auto-apply them.
FORBIDDEN_AUTO_PAIRS = frozenset(
    {
        ("vegetation", "tree_trunk"),
        ("tree", "tree_trunk"),
        ("potted plant", "tree_trunk"),
        ("building", "wall"),
        ("fence", "barrier"),
        ("street light", "pole"),
        ("streetlight", "pole"),
        ("traffic sign", "pole"),
        ("open", "open_doorway"),
        ("semi-open", "open_doorway"),
        ("semi open", "open_doorway"),
        ("stairs", "stairs_up"),
        ("stairs", "stairs_down"),
        ("stair", "stairs_up"),
        ("stair", "stairs_down"),
        ("step", "stairs_up"),
        ("step", "stairs_down"),
        ("step", "curb_up"),
        ("step", "curb_down"),
        ("curb", "curb_up"),
        ("curb", "curb_down"),
        ("kerb", "curb_up"),
        ("kerb", "curb_down"),
    }
)


class MappingError(ValueError):
    """Raised when a mapping would invent an unsafe label."""


@dataclass(frozen=True)
class MappingDecision:
    original_class: str
    proposed_class: str | None
    mapped_class: str | None
    mapping_type: str
    review_status: str
    taxonomy: str
    notes: str
    auto: bool


def normalize_class_name(name: str) -> str:
    """Lowercase, collapse separators, strip."""
    cleaned = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def is_directional_source(name: str) -> bool:
    return normalize_class_name(name) in DIRECTIONAL_SOURCE_CLASSES


def assert_auto_mapping_allowed(original_class: str, target: str | None) -> None:
    """Refuse auto-assignment of unsafe pairs."""
    if target is None:
        return
    original = normalize_class_name(original_class)
    mapped = target.strip()
    if (original, mapped) in FORBIDDEN_AUTO_PAIRS:
        raise MappingError(
            f"Refusing automatic mapping {original_class!r} → {target!r}. "
            "This pair requires human review and must not be invented."
        )
    if original in DIRECTIONAL_SOURCE_CLASSES and mapped in DIRECTIONAL_TARGETS:
        raise MappingError(
            f"Refusing automatic directional mapping {original_class!r} → {target!r}."
        )


def decide_mapping(
    source: SourceSpec,
    original_class: str,
    *,
    extra_attributes: dict[str, str] | None = None,
) -> MappingDecision:
    """Map one source class using the manifest plus hard safety rules."""
    original_norm = normalize_class_name(original_class)
    spec = _lookup_spec(source, original_norm)
    attributes = extra_attributes or {}

    step_type = normalize_class_name(attributes.get("type", ""))
    is_stairs = original_norm in {"stairs", "stair"}
    is_curb = step_type == "curb" or original_norm in {"curb", "kerb"}
    if is_stairs:
        return MappingDecision(
            original_class=original_class,
            proposed_class="stairs",
            mapped_class="stairs",
            mapping_type="DIRECT",
            review_status="APPROVED",
            taxonomy="custom",
            notes=(
                f"Taxonomy {TAXONOMY_VERSION} semantic mapping: source "
                f"{original_class!r} → stairs. Direction is not a detector label."
            ),
            auto=True,
        )
    if is_curb:
        return MappingDecision(
            original_class=original_class,
            proposed_class="curb",
            mapped_class="curb",
            mapping_type="DIRECT",
            review_status="APPROVED",
            taxonomy="custom",
            notes=(
                f"Taxonomy {TAXONOMY_VERSION} semantic mapping: source "
                f"{original_class!r} type={step_type or 'curb'} → curb. "
                "Direction is not a detector label."
            ),
            auto=True,
        )
    if original_norm in {"step", "steps"}:
        reason = f"step type={step_type}" if step_type else "generic step"
        return MappingDecision(
            original_class=original_class,
            proposed_class=None,
            mapped_class=None,
            mapping_type="CUSTOM",
            review_status="UNMAPPED_SOURCE_CLASS",
            taxonomy="custom",
            notes=(
                f"UNMAPPED_SOURCE_CLASS ({reason}). Not stairs or curb; "
                "left unmapped."
            ),
            auto=False,
        )

    if spec is None:
        if original_class in CUSTOM_CLASSES:
            return MappingDecision(
                original_class=original_class,
                proposed_class=original_class,
                mapped_class=original_class,
                mapping_type="CUSTOM",
                review_status="PENDING",
                taxonomy="custom",
                notes="Already a frozen Li-Stick class; sample remains PENDING until review.",
                auto=False,
            )
        return MappingDecision(
            original_class=original_class,
            proposed_class=None,
            mapped_class=None,
            mapping_type="CUSTOM",
            review_status="UNMAPPED_SOURCE_CLASS",
            taxonomy="custom",
            notes="Unlisted source class; left unmapped.",
            auto=False,
        )

    if spec.mapping_type not in MAPPING_TYPES:
        raise MappingError(f"Unknown mapping_type {spec.mapping_type!r}.")

    if spec.mapping_type == "MANUAL_DIRECTION" or is_directional_source(original_class):
        if spec.auto or spec.mapping_type != "MANUAL_DIRECTION":
            assert_auto_mapping_allowed(original_class, spec.target)
        return MappingDecision(
            original_class=original_class,
            proposed_class=None,
            mapped_class=None,
            mapping_type="CUSTOM",
            review_status="UNMAPPED_SOURCE_CLASS",
            taxonomy="custom",
            notes=(
                "UNMAPPED_SOURCE_CLASS. Not a Li-Stick detector class; "
                "direction was not guessed."
            ),
            auto=False,
        )

    if spec.target:
        _validate_target(spec.target, spec.taxonomy)
        if spec.auto:
            assert_auto_mapping_allowed(original_class, spec.target)

    if spec.mapping_type == "SEMANTIC_REVIEW" or not spec.auto:
        if spec.target is None:
            return MappingDecision(
                original_class=original_class,
                proposed_class=None,
                mapped_class=None,
                mapping_type=spec.mapping_type,
                review_status="UNMAPPED_SOURCE_CLASS",
                taxonomy=spec.taxonomy,
                notes=(
                    f"UNMAPPED_SOURCE_CLASS ({original_class}). "
                    "No semantically correct Li-Stick class."
                ),
                auto=False,
            )
        return MappingDecision(
            original_class=original_class,
            proposed_class=spec.target,
            mapped_class=None,
            mapping_type=spec.mapping_type,
            review_status="PENDING",
            taxonomy=spec.taxonomy,
            notes=spec.review or "Requires human review before promotion.",
            auto=False,
        )

    # DIRECT + auto: still PENDING until a reviewer accepts the sample.
    return MappingDecision(
        original_class=original_class,
        proposed_class=spec.target,
        mapped_class=spec.target,
        mapping_type=spec.mapping_type,
        review_status="PENDING",
        taxonomy=spec.taxonomy,
        notes="Direct mapping candidate; sample remains PENDING until review.",
        auto=True,
    )


def _lookup_spec(source: SourceSpec, original_norm: str) -> MappingSpec | None:
    for key, spec in source.listick_mapping.items():
        if normalize_class_name(key) == original_norm:
            return spec
    return None


def _validate_target(target: str, taxonomy: str) -> None:
    if taxonomy == "combined":
        allowed = COMBINED_CLASSES
    else:
        allowed = CUSTOM_CLASSES
    if target not in allowed:
        raise MappingError(
            f"Target class {target!r} is not in the {taxonomy} Li-Stick taxonomy."
        )
