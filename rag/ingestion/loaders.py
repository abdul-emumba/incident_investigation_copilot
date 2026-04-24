import json
import csv
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "rag"


def load_incident_tickets() -> List[Dict[str, Any]]:
    path = DATA_DIR / "incident_tickets.json"
    with open(path) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("tickets", list(data.values()))
    logger.info("Loaded %d incident tickets", len(records))
    return records


def load_production_logs() -> List[Dict[str, Any]]:
    path = DATA_DIR / "production_logs.json"
    with open(path) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("logs", list(data.values()))
    logger.info("Loaded %d production log entries", len(records))
    return records


def load_deployment_records() -> List[Dict[str, Any]]:
    path = DATA_DIR / "deployment_records.json"
    with open(path) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("deployments", list(data.values()))
    logger.info("Loaded %d deployment records", len(records))
    return records


def load_runbooks() -> List[Dict[str, Any]]:
    runbook_dir = DATA_DIR / "runbooks"
    runbooks = []
    for path in sorted(runbook_dir.glob("*.json")):
        with open(path) as f:
            rb = json.load(f)
        runbooks.append(rb)
    logger.info("Loaded %d runbook files", len(runbooks))
    return runbooks


def load_incident_event_log() -> List[Dict[str, Any]]:
    path = DATA_DIR / "incident_event_log.csv"
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    logger.info("Loaded %d incident event log rows", len(rows))
    return rows


def load_incident_responses() -> str:
    path = DATA_DIR / "incident_responses.txt"
    text = path.read_text()
    logger.info("Loaded incident_responses.txt (%d chars)", len(text))
    return text
