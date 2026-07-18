"""Small deterministic in-memory store for dated ticker memories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from math import exp, isfinite
from typing import Any


class MemoryLayer(StrEnum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"


_DEFAULT_CAPACITIES = {MemoryLayer.SHORT: 24, MemoryLayer.MID: 16, MemoryLayer.LONG: 8}
_HALF_LIVES = {MemoryLayer.SHORT: 7, MemoryLayer.MID: 30, MemoryLayer.LONG: 120}


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    event_date: date
    available_date: date
    layer: MemoryLayer
    ticker: str
    summary: str
    importance: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 200:
            raise ValueError("id must be a bounded nonblank string")
        if not isinstance(self.event_date, date) or not isinstance(self.available_date, date):
            raise TypeError("event_date and available_date must be dates")
        if self.available_date < self.event_date:
            raise ValueError("available_date must not be before event_date")
        if not isinstance(self.layer, MemoryLayer):
            raise TypeError("layer must be a MemoryLayer")
        if not isinstance(self.ticker, str) or not self.ticker.strip() or len(self.ticker) > 20:
            raise ValueError("ticker must be a bounded nonblank string")
        if (
            not isinstance(self.summary, str)
            or not self.summary.strip()
            or len(self.summary) > 2_000
        ):
            raise ValueError("summary must be a bounded nonblank string")
        if (
            isinstance(self.importance, bool)
            or not isinstance(self.importance, int | float)
            or not isfinite(self.importance)
            or not 0 <= self.importance <= 1
        ):
            raise ValueError("importance must be a finite number in [0, 1]")
        object.__setattr__(self, "importance", float(self.importance))

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_date": self.event_date.isoformat(),
            "available_date": self.available_date.isoformat(),
            "layer": self.layer.value,
            "ticker": self.ticker,
            "summary": self.summary,
            "importance": self.importance,
        }

    @classmethod
    def from_snapshot(cls, value: Mapping[str, object]) -> MemoryRecord:
        return cls(
            id=value["id"],  # type: ignore[arg-type]
            event_date=date.fromisoformat(value["event_date"]),  # type: ignore[arg-type]
            available_date=date.fromisoformat(value["available_date"]),  # type: ignore[arg-type]
            layer=MemoryLayer(value["layer"]),  # type: ignore[arg-type]
            ticker=value["ticker"],  # type: ignore[arg-type]
            summary=value["summary"],  # type: ignore[arg-type]
            importance=value["importance"],  # type: ignore[arg-type]
        )


class MemoryBook:
    """Bounded records with deterministic eviction and point-in-time retrieval."""

    def __init__(self, *, capacities: Mapping[MemoryLayer, int] | None = None) -> None:
        values = dict(_DEFAULT_CAPACITIES)
        if capacities is not None:
            for layer, capacity in capacities.items():
                if (
                    not isinstance(layer, MemoryLayer)
                    or isinstance(capacity, bool)
                    or not isinstance(capacity, int)
                    or capacity < 1
                ):
                    raise ValueError("capacities must map memory layers to positive integers")
                values[layer] = capacity
        self._capacities = values
        self._records: dict[MemoryLayer, list[MemoryRecord]] = {layer: [] for layer in MemoryLayer}

    def add(self, record: MemoryRecord) -> None:
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        if any(item.id == record.id for values in self._records.values() for item in values):
            raise ValueError("memory id must be unique")
        values = self._records[record.layer]
        values.append(record)
        if len(values) > self._capacities[record.layer]:
            values.remove(min(values, key=lambda item: (item.importance, item.id)))

    def retrieve(self, ticker: str, as_of: date) -> dict[MemoryLayer, tuple[MemoryRecord, ...]]:
        if not isinstance(ticker, str) or not ticker.strip() or not isinstance(as_of, date):
            raise ValueError("ticker and as_of are required")
        result: dict[MemoryLayer, tuple[MemoryRecord, ...]] = {}
        for layer in MemoryLayer:
            half_life = _HALF_LIVES[layer]
            eligible = [
                item
                for item in self._records[layer]
                if item.ticker == ticker and item.available_date <= as_of
            ]
            ranked = sorted(
                eligible,
                key=lambda item: (
                    -item.importance * exp(-(as_of - item.event_date).days / half_life),
                    item.id,
                ),
            )
            result[layer] = tuple(ranked[:3])
        return result

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            item.model_dump()
            for layer in MemoryLayer
            for item in sorted(self._records[layer], key=lambda value: value.id)
        ]

    @classmethod
    def load_snapshot(cls, snapshot: object) -> MemoryBook:
        if not isinstance(snapshot, list):
            raise TypeError("snapshot must be a list")
        book = cls()
        for value in snapshot:
            if not isinstance(value, Mapping):
                raise ValueError("snapshot records must be objects")
            book.add(MemoryRecord.from_snapshot(value))
        return book
