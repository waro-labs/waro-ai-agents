from dataclasses import dataclass
from typing import Any

from app.tools.sanitize import sanitize_value


@dataclass(frozen=True)
class ShapeProfile:
    entity: str
    allowed_rank_fields: frozenset[str]
    default_rank_fields: tuple[str, ...]
    active_fields: tuple[str, ...] = ()
    max_limit: int = 50


@dataclass(frozen=True)
class ShapeResult:
    rows: list[dict[str, Any]]
    execution: dict[str, Any]


class DataShaper:
    """Apply reusable, auditable analysis operations to structured tool rows."""

    def shape_ranked_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        profile: ShapeProfile,
        operations: list[dict[str, Any]] | None = None,
        sort_field: str | None = None,
        limit: int = 10,
    ) -> ShapeResult:
        normalized_operations = self.normalize_operations(operations)
        sort_fields = self.rank_fields(
            profile=profile,
            operations=normalized_operations,
            sort_field=sort_field,
        )
        active_rows = [
            row for row in rows if self._row_has_activity(row, profile.active_fields)
        ]
        ranked = sorted(
            active_rows,
            key=lambda row: tuple(
                self._numeric_value(row.get(field)) or 0 for field in sort_fields
            ),
            reverse=True,
        )
        bounded_limit = self._bounded_limit(limit, profile.max_limit)
        shaped = ranked[:bounded_limit]
        execution = {
            "entity": profile.entity,
            "input_rows": len(rows),
            "output_rows": len(shaped),
            "filtered_rows": len(rows) - len(active_rows),
            "sort_fields": list(sort_fields),
            "limit": bounded_limit,
            "operations": normalized_operations
            or self.default_rank_operations(
                profile=profile,
                sort_fields=sort_fields,
                limit=bounded_limit,
            ),
        }
        return ShapeResult(rows=sanitize_value(shaped), execution=sanitize_value(execution))

    def normalize_operations(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        allowed_types = {"filter", "rank", "sort", "limit", "compare", "group", "aggregate"}
        allowed_directions = {"asc", "desc"}
        operations: list[dict[str, Any]] = []
        for item in value[:8]:
            if not isinstance(item, dict) or item.get("type") not in allowed_types:
                continue
            operation: dict[str, Any] = {"type": str(item["type"])}
            if isinstance(item.get("condition"), str):
                operation["condition"] = item["condition"][:160]
            if isinstance(item.get("by"), list):
                operation["by"] = [
                    str(field)
                    for field in item["by"][:5]
                    if isinstance(field, str) and field
                ]
            elif isinstance(item.get("by"), str):
                operation["by"] = [str(item["by"])]
            if item.get("direction") in allowed_directions:
                operation["direction"] = str(item["direction"])
            if item.get("field") and isinstance(item.get("field"), str):
                operation["field"] = str(item["field"])
            if "value" in item:
                try:
                    operation["value"] = max(1, min(int(item["value"]), 50))
                except (TypeError, ValueError):
                    pass
            operations.append(operation)
        return operations

    def rank_fields(
        self,
        *,
        profile: ShapeProfile,
        operations: list[dict[str, Any]],
        sort_field: str | None,
    ) -> tuple[str, ...]:
        for operation in operations:
            if operation.get("type") not in {"rank", "sort"}:
                continue
            fields = operation.get("by")
            if not isinstance(fields, list):
                continue
            selected = tuple(
                str(field)
                for field in fields
                if isinstance(field, str) and field in profile.allowed_rank_fields
            )
            if selected:
                return selected
        if sort_field in profile.allowed_rank_fields:
            return self._prioritize_field(str(sort_field), profile.default_rank_fields)
        return profile.default_rank_fields

    def default_rank_operations(
        self,
        *,
        profile: ShapeProfile,
        sort_fields: tuple[str, ...],
        limit: int,
    ) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = []
        if profile.active_fields:
            condition = " OR ".join(f"{field} > 0" for field in profile.active_fields)
            operations.append({"type": "filter", "condition": condition})
        operations.append({"type": "rank", "by": list(sort_fields), "direction": "desc"})
        operations.append({"type": "limit", "value": self._bounded_limit(limit, profile.max_limit)})
        return operations

    def _prioritize_field(
        self,
        field: str,
        default_fields: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(dict.fromkeys((field, *default_fields)))

    def _row_has_activity(self, row: dict[str, Any], active_fields: tuple[str, ...]) -> bool:
        if not active_fields:
            return True
        return any((self._numeric_value(row.get(field)) or 0) > 0 for field in active_fields)

    def _bounded_limit(self, value: Any, max_limit: int) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError):
            limit = 10
        return max(1, min(limit, max_limit))

    def _numeric_value(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
