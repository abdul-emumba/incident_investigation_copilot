"""
Incident Investigation Copilot – Streamlit Dashboard

Run from the project root:
    streamlit run agents/dashboard.py
"""
from __future__ import annotations
from typing import Any

import math
import sys
from pathlib import Path

# ── Path bootstrap (must come before any local imports) ───────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PROJECT_ROOT / "rag"
for _p in (str(_PROJECT_ROOT), str(_RAG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from agents.models import AffectedService, TimelineEvent  # noqa: E402
from agents.orchestrator import InvestigationResult, Orchestrator  # noqa: E402
from agents.streaming import AGENT_DESCRIPTIONS, AGENT_LABELS, PIPELINE_AGENTS, StreamEvent  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Incident Investigation Copilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Root cause block */
.root-cause-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2440 100%);
    border-left: 5px solid #3B82F6;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    color: #f0f4ff;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
/* Evidence cards */
.evidence-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
}
/* Action item */
.action-item {
    background: #f0fdf4;
    border-left: 3px solid #10B981;
    border-radius: 4px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.92rem;
}
/* Conflict warning */
.conflict-item {
    background: #fff7ed;
    border-left: 3px solid #F59E0B;
    border-radius: 4px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
}
/* Agent status pill */
.agent-pill-idle    { color: #94a3b8; }
.agent-pill-running { color: #3B82F6; font-weight: 600; }
.agent-pill-done    { color: #10B981; font-weight: 600; }
.agent-pill-error   { color: #EF4444; font-weight: 600; }
/* Memory incident card */
.memory-card {
    background: #f8f4ff;
    border: 1px solid #ddd6fe;
    border-left: 4px solid #7C3AED;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.8rem;
}
.memory-card-header {
    font-weight: 600;
    color: #5B21B6;
    margin-bottom: 0.3rem;
    font-size: 0.95rem;
}
.memory-insight {
    background: #ede9fe;
    border-left: 3px solid #7C3AED;
    border-radius: 4px;
    padding: 0.45rem 0.85rem;
    margin-bottom: 0.35rem;
    font-size: 0.9rem;
    color: #4C1D95;
}
/* Sidebar subheading */
.sidebar-section { font-size: 0.75rem; text-transform: uppercase;
                   letter-spacing: 0.08em; color: #94a3b8; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached orchestrator ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _timeline_chart(events: list[TimelineEvent]) -> go.Figure | None:
    known = [e for e in events if e.timestamp not in ("unknown", None, "")]
    if not known:
        return None

    rows = []
    for e in known:
        rows.append({
            "Time": e.timestamp,
            "Category": e.event_type.replace("_", " ").title(),
            "Description": (e.description[:72] + "…") if len(e.description) > 72 else e.description,
            "Service": e.service or "—",
            "Source": e.source,
        })
    df = pd.DataFrame(rows)
    try:
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
        df = df.dropna(subset=["Time"]).sort_values("Time")
    except Exception:
        pass

    color_map = {
        "Deployment":    "#3B82F6",
        "Error":         "#EF4444",
        "Alert":         "#F59E0B",
        "Ticket":        "#8B5CF6",
        "Recovery":      "#10B981",
        "Config Change": "#6B7280",
        "Other":         "#94a3b8",
    }

    categories = df["Category"].unique().tolist()
    fig = go.Figure()
    for cat in categories:
        sub = df[df["Category"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["Time"],
            y=[cat] * len(sub),
            mode="markers",
            name=cat,
            marker=dict(
                size=16,
                color=color_map.get(cat, "#94a3b8"),
                line=dict(width=2, color="white"),
                symbol="circle",
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Service: %{customdata[1]}<br>"
                "Source: %{customdata[2]}<br>"
                "Time: %{x}<extra></extra>"
            ),
            customdata=sub[["Description", "Service", "Source"]].values,
        ))
        # Connecting line for each category
        fig.add_trace(go.Scatter(
            x=sub["Time"], y=[cat] * len(sub),
            mode="lines",
            line=dict(color=color_map.get(cat, "#94a3b8"), width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=max(260, 60 * len(categories) + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title=""),
        yaxis=dict(showgrid=False, title="", categoryorder="array",
                   categoryarray=list(reversed(categories))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def _services_chart(affected: list[AffectedService], critical_path: list[str]) -> go.Figure | None:
    if not affected:
        return None

    role_color = {
        "root_cause_candidate": "#EF4444",
        "directly_affected":    "#F59E0B",
        "transitively_affected":"#3B82F6",
    }
    role_label = {
        "root_cause_candidate": "Root Cause Candidate",
        "directly_affected":    "Directly Affected",
        "transitively_affected":"Transitively Affected",
    }

    n = len(affected)
    xs, ys, texts, colors, hovers = [], [], [], [], []

    for i, svc in enumerate(affected):
        angle = 2 * math.pi * i / n - math.pi / 2
        r = max(0.5, svc.depth * 0.8) if svc.depth > 0 else 0
        xs.append(r * math.cos(angle))
        ys.append(r * math.sin(angle))
        texts.append(svc.service_id)
        colors.append(role_color.get(svc.role, "#94a3b8"))
        cb = "✓" if svc.has_circuit_breaker else "✗"
        fb = "✓" if svc.has_fallback else "✗"
        hovers.append(
            f"<b>{svc.service_id}</b><br>"
            f"Role: {role_label.get(svc.role, svc.role)}<br>"
            f"Depth: {svc.depth}<br>"
            f"Circuit breaker: {cb}  Fallback: {fb}<br>"
            f"Team: {svc.team or '—'}"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(size=22, color=colors, line=dict(width=2, color="white")),
        text=texts,
        textposition="top center",
        hovertext=hovers,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    # Legend via invisible dummy traces
    for role, color in role_color.items():
        if any(s.role == role for s in affected):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color),
                name=role_label[role],
            ))

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def _critical_path_chart(path: list[str]) -> go.Figure | None:
    if len(path) < 2:
        return None

    n = len(path)
    xs = [i / (n - 1) for i in range(n)]
    colors = ["#EF4444"] + ["#F59E0B"] * (n - 2) + ["#94a3b8"]

    fig = go.Figure()
    # Connecting line
    fig.add_trace(go.Scatter(
        x=xs, y=[0.5] * n,
        mode="lines",
        line=dict(color="#cbd5e1", width=3),
        showlegend=False, hoverinfo="skip",
    ))
    # Nodes
    fig.add_trace(go.Scatter(
        x=xs, y=[0.5] * n,
        mode="markers+text",
        marker=dict(size=28, color=colors, line=dict(width=2, color="white")),
        text=path,
        textposition="bottom center",
        hoverinfo="text",
        showlegend=False,
    ))
    # Arrows between nodes
    for i in range(n - 1):
        fig.add_annotation(
            x=xs[i + 1], y=0.5, ax=xs[i], ay=0.5,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="#94a3b8", arrowwidth=2,
        )

    fig.update_layout(
        height=140,
        margin=dict(l=20, r=20, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.1, 0.9]),
    )
    return fig


def _confidence_gauge(score: float) -> go.Figure:
    pct = round(score * 100)
    if score >= 0.75:
        bar_color, label = "#10B981", "High"
    elif score >= 0.50:
        bar_color, label = "#F59E0B", "Medium"
    else:
        bar_color, label = "#EF4444", "Low"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": f"Confidence · {label}", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 100],"color": "#D1FAE5"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.75, "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Result display ────────────────────────────────────────────────────────────

def _show_results(result: InvestigationResult, orc: Orchestrator, context: dict) -> None:
    r = result.report
    if r is None:
        st.error("Investigation completed but no report was generated.")
        if result.errors:
            for e in result.errors:
                st.caption(f"↳ {e}")
        return

    tab_overview, tab_timeline, tab_services, tab_evidence, tab_memory, tab_debug = st.tabs([
        "📋 Overview", "🕐 Timeline", "🔗 Services", "📎 Evidence", "🧠 Memory", "🛠 Debug",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_overview:
        col_rc, col_gauge = st.columns([3, 1])
        with col_rc:
            st.subheader("Root Cause")
            st.markdown(
                f'<div class="root-cause-box">{r.root_cause}</div>',
                unsafe_allow_html=True,
            )
        with col_gauge:
            st.plotly_chart(_confidence_gauge(r.confidence_score), use_container_width=True)

        st.subheader("AI Narrative")
        with st.spinner("Streaming investigation narrative…"):
            st.write_stream(orc.report_generator.stream_narrative(context))

        if r.recommended_actions:
            st.subheader("Recommended Actions")
            for act in r.recommended_actions:
                st.markdown(
                    f'<div class="action-item">▶ {act}</div>',
                    unsafe_allow_html=True,
                )

        if r.conflicting_signals:
            st.subheader("⚠ Conflicting Signals")
            for sig in r.conflicting_signals:
                st.markdown(
                    f'<div class="conflict-item">⚠ {sig}</div>',
                    unsafe_allow_html=True,
                )

    # ── Timeline ──────────────────────────────────────────────────────────────
    with tab_timeline:
        if not r.timeline:
            st.info("No timeline events were extracted.")
        else:
            chart = _timeline_chart(r.timeline)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            st.subheader("All Events")
            rows = [
                {
                    "Timestamp": e.timestamp,
                    "Type": e.event_type.replace("_", " ").title(),
                    "Description": e.description,
                    "Service": e.service or "—",
                    "Source": e.source,
                }
                for e in r.timeline
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp":   st.column_config.TextColumn(width="medium"),
                    "Type":        st.column_config.TextColumn(width="small"),
                    "Description": st.column_config.TextColumn(width="large"),
                    "Service":     st.column_config.TextColumn(width="small"),
                    "Source":      st.column_config.TextColumn(width="small"),
                },
            )

    # ── Services ──────────────────────────────────────────────────────────────
    with tab_services:
        graph_res = result.agent_results.get("graph")
        affected = []
        critical_path: list[str] = []

        if graph_res and graph_res.success and graph_res.data:
            affected = graph_res.data.affected_services
            critical_path = graph_res.data.critical_path

        if critical_path and len(critical_path) >= 2:
            st.subheader("Critical Failure Path")
            cp_fig = _critical_path_chart(critical_path)
            if cp_fig:
                st.plotly_chart(cp_fig, use_container_width=True)

        if affected:
            col_chart, col_table = st.columns([1, 1])
            with col_chart:
                st.subheader("Blast Radius")
                svc_fig = _services_chart(affected, critical_path)
                if svc_fig:
                    st.plotly_chart(svc_fig, use_container_width=True)

            with col_table:
                st.subheader("Affected Services")
                svc_rows = [
                    {
                        "Service": s.service_id,
                        "Role": s.role.replace("_", " ").title(),
                        "Depth": s.depth,
                        "Circuit Breaker": "✓" if s.has_circuit_breaker else "✗",
                        "Fallback": "✓" if s.has_fallback else "✗",
                        "Team": s.team or "—",
                    }
                    for s in affected
                ]
                st.dataframe(
                    pd.DataFrame(svc_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            if graph_res.data.shared_dependencies:
                st.subheader("Shared Dependencies (Single Points of Failure)")
                for spof in graph_res.data.shared_dependencies:
                    st.markdown(f"- `{spof}`")
        else:
            st.info("No service dependency data available.")

        if not (graph_res and graph_res.data and graph_res.data.graph_available):
            st.caption("ℹ Graph analysis ran in RAG-fallback mode (Neo4j not connected).")

    # ── Evidence ──────────────────────────────────────────────────────────────
    with tab_evidence:
        if r.evidence:
            st.subheader("Evidence Citations")
            src_icon = {
                "log": "📋", "deployment": "🚀", "ticket": "🎫",
                "event": "📡", "runbook": "📖",
            }
            for ev in r.evidence:
                icon = next(
                    (v for k, v in src_icon.items() if k in ev.lower()),
                    "📌",
                )
                st.markdown(
                    f'<div class="evidence-card">{icon} {ev}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No evidence citations were extracted.")

        # Show raw evidence from individual agents in expanders
        st.subheader("Raw Agent Evidence")
        for agent_name in ("log_analysis", "timeline", "hypothesis"):
            agent_res = result.agent_results.get(agent_name)
            if agent_res and agent_res.success and agent_res.data:
                raw = getattr(agent_res.data, "raw_evidence", [])
                if raw:
                    with st.expander(f"{AGENT_LABELS.get(agent_name, agent_name)} — raw evidence"):
                        for piece in raw:
                            st.text(piece)
                            st.divider()

    # ── Memory ────────────────────────────────────────────────────────────────
    with tab_memory:
        mem_res = result.agent_results.get("memory")
        mem_data = mem_res.data if (mem_res and mem_res.success and mem_res.data) else None

        if mem_data and mem_data.similar_incidents:
            st.subheader("Pattern Insights from History")
            if mem_data.pattern_insights:
                for insight in mem_data.pattern_insights:
                    st.markdown(
                        f'<div class="memory-insight">💡 {insight}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No recurring patterns detected across similar incidents.")

            st.subheader(f"Similar Past Incidents ({len(mem_data.similar_incidents)} found)")
            for sim in mem_data.similar_incidents:
                similarity_pct = f"{sim.similarity_score * 100:.0f}%"
                confidence_pct = f"{sim.confidence * 100:.0f}%"
                services_str = ", ".join(f"`{s}`" for s in sim.affected_services) or "—"
                with st.expander(
                    f"🗂 {sim.incident_id}  ·  similarity {similarity_pct}  ·  confidence {confidence_pct}",
                    expanded=len(mem_data.similar_incidents) == 1,
                ):
                    st.markdown(
                        f'<div class="memory-card">'
                        f'<div class="memory-card-header">Root Cause</div>'
                        f'{sim.root_cause}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption("**Affected Services**")
                        st.markdown(services_str)
                    with col_b:
                        st.caption("**Recorded**")
                        ts = sim.timestamp[:19].replace("T", " ") if sim.timestamp else "—"
                        st.markdown(f"`{ts} UTC`")
                    if sim.investigation_summary:
                        st.caption("**Summary**")
                        st.markdown(sim.investigation_summary)
        else:
            st.info(
                "No similar past incidents found. "
                "After completing more investigations, patterns will appear here automatically."
            )

    # ── Debug ─────────────────────────────────────────────────────────────────
    with tab_debug:
        st.caption(f"Total latency: {result.total_latency_ms / 1000:.1f}s")
        if result.errors:
            st.warning(f"{len(result.errors)} non-fatal error(s):")
            for e in result.errors:
                st.caption(f"  ↳ {e}")

        for name in PIPELINE_AGENTS:
            agent_res = result.agent_results.get(name)
            if not agent_res:
                continue
            label = AGENT_LABELS.get(name, name)
            status = "✅" if agent_res.success else "❌"
            with st.expander(f"{status} {label}"):
                if agent_res.error:
                    st.error(agent_res.error)
                if agent_res.data is not None:
                    try:
                        import dataclasses, json
                        st.json(dataclasses.asdict(agent_res.data))
                    except Exception:
                        st.text(str(agent_res.data))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(agent_statuses: dict[str, str]) -> tuple[str, bool]:
    with st.sidebar:
        st.markdown("# 🔍 Incident Copilot")
        st.caption("Multi-Agent AI · RAG · Knowledge Graph")
        st.divider()

        query = st.text_area(
            "Investigation Query",
            placeholder=(
                "e.g. Why did auth-service fail on 2024-01-15 around 14:00?\n"
                "or: Investigate INC-0042"
            ),
            height=120,
            key="query_input",
        )
        submit = st.button("🔍 Investigate", type="primary", use_container_width=True)

        st.divider()
        st.markdown('<p class="sidebar-section">Agent Pipeline</p>', unsafe_allow_html=True)

        icon_map = {"idle": "⬜", "running": "⏳", "done": "✅", "error": "❌"}
        for name in PIPELINE_AGENTS:
            status = agent_statuses.get(name, "idle")
            icon = icon_map.get(status, "⬜")
            label = AGENT_LABELS.get(name, name)
            desc = AGENT_DESCRIPTIONS.get(name, "")
            st.markdown(f"{icon} **{label}**  \n<small>{desc}</small>", unsafe_allow_html=True)

    return query, submit


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Session state
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "context" not in st.session_state:
        st.session_state["context"] = {}
    if "agent_statuses" not in st.session_state:
        st.session_state["agent_statuses"] = {a: "idle" for a in PIPELINE_AGENTS}

    query, submit = _render_sidebar(st.session_state["agent_statuses"])

    st.markdown("## 🔍 Incident Investigation Copilot")
    st.caption("Powered by Multi-Agent AI · Hybrid RAG · Neo4j Knowledge Graph")

    if submit and query.strip():
        # Reset state for new investigation
        st.session_state["result"] = None
        st.session_state["context"] = {"query": query.strip()}
        st.session_state["agent_statuses"] = {a: "idle" for a in PIPELINE_AGENTS}

        with st.spinner("Initialising investigation system…"):
            orc = get_orchestrator()

        # ── Live agent pipeline progress ──────────────────────────────────────
        st.subheader("Investigation in Progress")
        progress_bar = st.progress(0, text="Starting…")
        agent_slots: dict[str, Any] = {}

        with st.container():
            cols = st.columns(len(PIPELINE_AGENTS))
            for i, name in enumerate(PIPELINE_AGENTS):
                with cols[i]:
                    agent_slots[name] = st.empty()
                    agent_slots[name].markdown(
                        f"⬜ **{AGENT_LABELS[name]}**",
                        help=AGENT_DESCRIPTIONS[name],
                    )

        completed = 0
        total = len(PIPELINE_AGENTS)
        context_accum: dict = {"query": query.strip()}

        for event in orc.stream_investigate(query.strip()):
            if event.kind == "agent_start":
                st.session_state["agent_statuses"][event.agent] = "running"
                agent_slots[event.agent].markdown(
                    f"⏳ **{AGENT_LABELS[event.agent]}**",
                    help=AGENT_DESCRIPTIONS.get(event.agent, ""),
                )
                progress_bar.progress(
                    completed / total,
                    text=f"Running {AGENT_LABELS.get(event.agent, event.agent)}…",
                )

            elif event.kind == "agent_done":
                completed += 1
                has_error = bool(event.error)
                st.session_state["agent_statuses"][event.agent] = "error" if has_error else "done"
                agent_slots[event.agent].markdown(
                    f"{'❌' if has_error else '✅'} **{AGENT_LABELS[event.agent]}**",
                    help=event.error or AGENT_DESCRIPTIONS.get(event.agent, ""),
                )
                progress_bar.progress(
                    completed / total,
                    text=f"Completed {completed}/{total} agents…",
                )
                # Accumulate context so stream_narrative() has all agent results
                if event.data:
                    context_accum[event.agent] = event.data
                    if event.agent == "planner" and event.data.success:
                        context_accum["plan"] = event.data.data

            elif event.kind == "complete":
                progress_bar.progress(1.0, text="Investigation complete ✅")
                st.session_state["result"] = event.data
                st.session_state["context"] = context_accum

        st.rerun()

    # ── Show cached results ───────────────────────────────────────────────────
    if st.session_state.get("result"):
        result: InvestigationResult = st.session_state["result"]
        orc = get_orchestrator()
        st.success(
            f"Investigation completed in {result.total_latency_ms / 1000:.1f}s "
            f"· {len(result.agent_results)} agents · "
            f"confidence {result.report.confidence_score:.0%}"
            if result.report else "Investigation completed.",
        )
        _show_results(result, orc, st.session_state["context"])

    elif not submit:
        # Landing state
        st.markdown("""
**How to use:**
1. Type your incident query in the sidebar — e.g. *"Investigate INC-0042"* or *"Why did auth-service fail after the deployment?"*
2. Click **Investigate** to start the multi-agent pipeline.
3. Watch each agent complete in real-time, then explore the results across four tabs.

**Agent pipeline:**

| Agent | Responsibility |
|---|---|
| Planner | Decomposes the query into investigation steps |
| Log Analysis | Detects anomalies and clusters related errors |
| Timeline | Builds the chronological event sequence |
| Graph | Traces cascading failures through the dependency graph |
| Hypothesis | Generates ranked root-cause explanations |
| Critic | Verifies claims and surfaces conflicting signals |
| Report Generator | Synthesises the final structured report |
""")
    elif submit and not query.strip():
        st.warning("Please enter an investigation query.")


if __name__ == "__main__":
    main()
