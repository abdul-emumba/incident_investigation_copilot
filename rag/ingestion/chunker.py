import logging
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chunk_incident_tickets(records: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for r in records:
        resolution = r.get("resolution") or "open"
        text = (
            f"{r.get('severity', 'UNKNOWN')} incident: {r.get('title', '')}\n"
            f"affected: {r.get('affected_service', '')}\n"
            f"description: {r.get('description', '')}\n"
            f"resolution: {resolution}"
        )
        metadata = {
            "source_dataset": "incident_tickets",
            "incident_id": r.get("incident_id"),
            "ticket_id": r.get("ticket_id"),
            "severity": r.get("severity"),
            "affected_service": r.get("affected_service"),
            "service": r.get("affected_service"),
            "timestamp": r.get("timestamp"),
            "has_resolution": bool(r.get("resolution")),
            "wrong_assumption": r.get("wrong_assumption"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    logger.info("Chunked %d incident ticket documents", len(docs))
    return docs


def chunk_production_logs(records: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for r in records:
        text = (
            f"{r.get('level', '')} {r.get('service', '')} — {r.get('message', '')}\n"
            f"request_id: {r.get('request_id', '')}\n"
            f"incident: {r.get('incident_id', 'N/A')}"
        )
        metadata = {
            "source_dataset": "production_logs",
            "timestamp": r.get("timestamp"),
            "service": r.get("service"),
            "level": r.get("level"),
            "incident_id": r.get("incident_id"),
            "request_id": r.get("request_id"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    logger.info("Chunked %d production log documents", len(docs))
    return docs


def chunk_deployment_records(records: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for r in records:
        breaking = r.get("breaking_change", "N/A")
        text = (
            f"Deployment {r.get('version', '')} to {r.get('service', '')} at {r.get('timestamp', '')}\n"
            f"environment: {r.get('environment', '')}   deployed_by: {r.get('deployed_by', '')}\n"
            f"change: {r.get('change_description', '')}\n"
            f"breaking: {breaking}"
        )
        metadata = {
            "source_dataset": "deployment_records",
            "deployment_id": r.get("deployment_id"),
            "service": r.get("service"),
            "version": r.get("version"),
            "timestamp": r.get("timestamp"),
            "environment": r.get("environment"),
            "breaking_change": bool(r.get("breaking_change")) if r.get("breaking_change") is not None else False,
            "rollback_available": bool(r.get("rollback_available")),
            "incident_ref": r.get("incident_ref"),
            "incident_id": r.get("incident_ref"),
            "tags": r.get("tags", []),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    logger.info("Chunked %d deployment record documents", len(docs))
    return docs


def chunk_runbooks(runbooks: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for rb in runbooks:
        service = rb.get("service", "")
        runbook_id = rb.get("runbook_id", "")
        for issue in rb.get("common_issues", []):
            issue_id = issue.get("issue_id", "")
            symptoms = " | ".join(issue.get("symptoms", []))
            causes = " | ".join(issue.get("likely_causes", []))
            escalation = issue.get("escalation", "")
            issue_examples = issue.get("incident_examples", rb.get("incident_examples", []))

            # Type A — issue-level chunk
            issue_text = (
                f"Runbook {service} — {issue_id}: {issue.get('title', '')}\n"
                f"symptoms: {symptoms}\n"
                f"likely causes: {causes}\n"
                f"escalation: {escalation}"
            )
            issue_meta = {
                "source_dataset": "runbooks",
                "service": service,
                "runbook_id": runbook_id,
                "issue_id": issue_id,
                "chunk_type": "issue",
                "incident_examples": issue_examples,
                "incident_id": None,
            }
            docs.append(Document(page_content=issue_text, metadata=issue_meta))

            # Type B — step-level chunks
            for step in issue.get("steps", []):
                step_text = (
                    f"Runbook {service} {issue_id} step {step.get('order', '')}: {step.get('action', '')}\n"
                    f"command: {step.get('command', '')}\n"
                    f"note: {step.get('note', '')}"
                )
                step_meta = {
                    "source_dataset": "runbooks",
                    "service": service,
                    "runbook_id": runbook_id,
                    "issue_id": issue_id,
                    "chunk_type": "step",
                    "incident_examples": issue_examples,
                    "incident_id": None,
                }
                docs.append(Document(page_content=step_text, metadata=step_meta))

    logger.info("Chunked %d runbook documents", len(docs))
    return docs


def chunk_incident_event_log(rows: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for row in rows:
        text = (
            f"{row.get('timestamp', '')} {row.get('actor', '')} — {row.get('event_type', '')}\n"
            f"incident: {row.get('incident_id', '')}   service: {row.get('service', '')}\n"
            f"deployment: {row.get('related_deployment_id', 'N/A')}"
        )
        metadata = {
            "source_dataset": "incident_event_log",
            "incident_id": row.get("incident_id"),
            "service": row.get("service"),
            "actor": row.get("actor"),
            "timestamp": row.get("timestamp"),
            "deployment_id": row.get("related_deployment_id") or None,
        }
        docs.append(Document(page_content=text, metadata=metadata))
    logger.info("Chunked %d incident event log documents", len(docs))
    return docs


def chunk_incident_responses(text: str) -> List[Document]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs = []
    for i, para in enumerate(paragraphs):
        metadata = {
            "source_dataset": "incident_responses",
            "source": "incident_responses",
            "chunk_index": i,
            "incident_id": None,
            "service": None,
        }
        docs.append(Document(page_content=para, metadata=metadata))
    logger.info("Chunked %d incident response paragraphs", len(docs))
    return docs


def build_all_chunks(
    tickets, logs, deployments, runbooks, event_log, responses_text
) -> Tuple[List[Document], Dict[str, int]]:
    all_docs = []
    counts: Dict[str, int] = {}

    ticket_docs = chunk_incident_tickets(tickets)
    counts["incident_tickets"] = len(ticket_docs)
    all_docs.extend(ticket_docs)

    log_docs = chunk_production_logs(logs)
    counts["production_logs"] = len(log_docs)
    all_docs.extend(log_docs)

    deploy_docs = chunk_deployment_records(deployments)
    counts["deployment_records"] = len(deploy_docs)
    all_docs.extend(deploy_docs)

    runbook_docs = chunk_runbooks(runbooks)
    counts["runbooks"] = len(runbook_docs)
    all_docs.extend(runbook_docs)

    event_docs = chunk_incident_event_log(event_log)
    counts["incident_event_log"] = len(event_docs)
    all_docs.extend(event_docs)

    response_docs = chunk_incident_responses(responses_text)
    counts["incident_responses"] = len(response_docs)
    all_docs.extend(response_docs)

    return all_docs, counts
