from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class ValidationReportError(ValueError):
    pass


@dataclass(frozen=True)
class ReportSection:
    title: str
    results: Mapping[str, Any]
    source: str

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.source.strip():
            raise ValidationReportError("report section requires title and result source")
        if not self.results:
            raise ValidationReportError("report section requires observed results")
        for key, value in self.results.items():
            _validate_observed(value, key)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationReportError(f"cannot read result artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationReportError(f"result artifact must contain an object: {path}")
    return value


def _validate_observed(value: Any, field: str) -> None:
    if value is None:
        raise ValidationReportError(f"result {field!r} has no observed value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationReportError(f"result {field!r} is not finite")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_observed(child, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_observed(child, f"{field}[{index}]")


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return f"`{json.dumps(value, sort_keys=True)}`"
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_section(section: ReportSection) -> list[str]:
    for key, value in section.results.items():
        _validate_observed(value, key)
    lines = [f"## {section.title}", "", f"Source: `{section.source}`", ""]
    for key, value in section.results.items():
        label = key.replace("_", " ").strip().title()
        lines.append(f"- **{label}:** {_render_value(value)}")
    lines.append("")
    return lines


def generate_validation_report(
    output_path: Path | str,
    *,
    manifest_path: Path | str,
    stats_path: Path | str | None = None,
    sections: Iterable[ReportSection] = (),
) -> Path:
    """Render only persisted results and explicitly sourced observed results."""
    manifest_source = Path(manifest_path)
    manifest = _load_json(manifest_source)
    required = (
        "dataset",
        "dataset_name",
        "files",
        "source_files",
        "source_fingerprint",
        "synthetic",
        "sampled",
        "locale",
        "currency",
        "total_records",
        "catalog_customers",
        "catalog_articles",
        "interacting_customers",
        "interacting_articles",
        "purchases",
        "date_range",
        "semantics",
    )
    missing = set(required) - set(manifest)
    if missing:
        raise ValidationReportError(
            f"manifest lacks measured fields: {', '.join(sorted(missing))}"
        )
    dataset_results = {key: manifest[key] for key in required}
    ordered_sections = [
        ReportSection("Dataset", dataset_results, str(manifest_source))
    ]

    if stats_path is not None:
        stats_source = Path(stats_path)
        stats = _load_json(stats_source)
        if not stats:
            raise ValidationReportError("stats artifact contains no measured results")
        ordered_sections.append(ReportSection("EDA", stats, str(stats_source)))
    ordered_sections.extend(sections)

    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Validation Report",
        "",
        f"Generated from result artifacts at `{generated_at}`.",
        "",
    ]
    for section in ordered_sections:
        lines.extend(_render_section(section))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    os.replace(temporary, output)
    return output
