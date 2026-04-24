from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import IncidentReport, SimilarIncident

_MEMORY_FILE = Path(__file__).resolve().parent.parent / "data" / "incident_memory.json"


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, filtering short/stop words."""
    _STOP = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
              "or", "is", "was", "are", "were", "be", "been", "being",
              "have", "has", "had", "do", "did", "does", "not", "with",
              "by", "from", "this", "that", "it", "as", "why", "how",
              "what", "when", "which", "who"}
    tokens = re.findall(r"[a-z0-9][-a-z0-9]*", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class IncidentMemoryStore:
    """JSON-backed store for completed incident investigations."""

    def __init__(self) -> None:
        _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if _MEMORY_FILE.exists():
            try:
                return json.loads(_MEMORY_FILE.read_text())
            except Exception:
                return []
        return []

    def _save(self) -> None:
        _MEMORY_FILE.write_text(json.dumps(self._entries, indent=2))

    def save_incident(self, report: IncidentReport, query: str) -> None:
        entry: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "incident_id": report.incident_id or f"INC-{uuid.uuid4().hex[:6].upper()}",
            "query": query,
            "root_cause": report.root_cause,
            "affected_services": report.affected_services,
            "confidence": report.confidence_score,
            "investigation_summary": report.investigation_summary,
            "recommended_actions": report.recommended_actions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Avoid duplicating the same query+root_cause
        existing_keys = {(e["query"], e["root_cause"]) for e in self._entries}
        if (entry["query"], entry["root_cause"]) not in existing_keys:
            self._entries.append(entry)
            self._save()

    def find_similar(self, query: str, top_k: int = 3) -> list[SimilarIncident]:
        if not self._entries:
            return []

        query_tokens = _tokenize(query)
        scored: list[tuple[float, dict]] = []

        for entry in self._entries:
            candidate_tokens = _tokenize(
                entry["query"] + " " + entry["root_cause"] + " " +
                " ".join(entry.get("affected_services", []))
            )
            score = _jaccard(query_tokens, candidate_tokens)
            if score > 0.05:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SimilarIncident] = []
        for score, e in scored[:top_k]:
            results.append(SimilarIncident(
                incident_id=e.get("incident_id", "unknown"),
                query=e["query"],
                root_cause=e["root_cause"],
                affected_services=e.get("affected_services", []),
                confidence=e.get("confidence", 0.0),
                similarity_score=round(score, 3),
                investigation_summary=e.get("investigation_summary", ""),
                timestamp=e.get("timestamp", ""),
            ))
        return results

    def count(self) -> int:
        return len(self._entries)
