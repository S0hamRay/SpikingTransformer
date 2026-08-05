"""Knowledge-graph and Leiden-community visualization helpers for Gradio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from db.neo4j_store import Neo4jGraphStore

# Skip Chunk flood by default — papers/authors/concepts carry the structure.
DEFAULT_LABELS = ("Paper", "Author", "Concept", "Community", "Dataset", "Venue")
REL_TYPES_KG = (
    "AUTHORED_BY",
    "USES_METHOD",
    "MENTIONS",
    "EVALUATES_ON",
    "REPORTS_METRIC",
    "PUBLISHED_IN",
    "BELONGS_TO",
    "CITES",
    "EXTENDS",
    "RELATED_TO",
)
LABEL_COLORS = {
    "Paper": "#2563eb",
    "Author": "#16a34a",
    "Concept": "#d97706",
    "Community": "#7c3aed",
    "Dataset": "#0891b2",
    "Venue": "#64748b",
    "Chunk": "#94a3b8",
    "Institution": "#a855f7",
}
COMMUNITY_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#059669",
    "#ea580c",
]


@dataclass
class GraphSnapshot:
    """Lightweight graph payload for plotting."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _node_key(labels: list[str], props: dict[str, Any]) -> str:
    label = labels[0] if labels else "Node"
    for field_name in ("id", "leidenId", "name", "title"):
        if props.get(field_name) is not None:
            return f"{label}:{props[field_name]}"
    return f"{label}:{id(props)}"


def _node_label(labels: list[str], props: dict[str, Any]) -> str:
    label = labels[0] if labels else "Node"
    if label == "Paper":
        title = str(props.get("title") or props.get("source_filename") or "Paper")
        return title if len(title) <= 42 else title[:39] + "…"
    if label == "Community":
        lid = props.get("leidenId")
        size = props.get("size")
        return f"Community {lid}" + (f" (n={size})" if size is not None else "")
    name = props.get("name") or props.get("title")
    if name:
        text = str(name)
        return text if len(text) <= 36 else text[:33] + "…"
    return label


def fetch_knowledge_graph(
    *,
    store: Neo4jGraphStore | None = None,
    max_nodes: int = 80,
    include_chunks: bool = False,
) -> GraphSnapshot:
    """Fetch a Paper/Author/Concept-centric subgraph from Neo4j."""
    owned = store is None
    store = store or Neo4jGraphStore()
    labels = list(DEFAULT_LABELS)
    if include_chunks:
        labels.append("Chunk")

    try:
        with store.connect().session() as session:
            counts = {
                row["l"]: int(row["c"])
                for row in session.run(
                    "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c"
                )
                if row["l"]
            }
            community_n = int(
                session.run("MATCH (c:Community) RETURN count(c) AS n").single()["n"]
            )

            # Prefer structural edges; Mentions via Chunk are collapsed conceptually
            # by also pulling USES_METHOD / paper-level links.
            rows = list(
                session.run(
                    """
                    MATCH (a)-[r]->(b)
                    WHERE type(r) IN $rels
                      AND any(l IN labels(a) WHERE l IN $labels)
                      AND any(l IN labels(b) WHERE l IN $labels)
                    RETURN labels(a) AS a_labels, properties(a) AS a_props,
                           type(r) AS rel,
                           labels(b) AS b_labels, properties(b) AS b_props,
                           properties(r) AS r_props
                    LIMIT $limit
                    """,
                    rels=list(REL_TYPES_KG),
                    labels=labels,
                    limit=max(50, max_nodes * 4),
                )
            )

            # Also pull Concept co-occurrence when present (useful even before Leiden).
            co_rows = list(
                session.run(
                    """
                    MATCH (a:Concept)-[r:CO_OCCURS_WITH]->(b:Concept)
                    RETURN labels(a) AS a_labels, properties(a) AS a_props,
                           type(r) AS rel,
                           labels(b) AS b_labels, properties(b) AS b_props,
                           properties(r) AS r_props
                    LIMIT $limit
                    """,
                    limit=max(20, max_nodes),
                )
            )
            rows.extend(co_rows)

            nodes: dict[str, dict[str, Any]] = {}
            edges: list[dict[str, Any]] = []
            for row in rows:
                a_labels = list(row["a_labels"] or [])
                b_labels = list(row["b_labels"] or [])
                a_props = dict(row["a_props"] or {})
                b_props = dict(row["b_props"] or {})
                r_props = dict(row["r_props"] or {})
                a_key = _node_key(a_labels, a_props)
                b_key = _node_key(b_labels, b_props)
                if a_key not in nodes:
                    nodes[a_key] = {
                        "id": a_key,
                        "label": _node_label(a_labels, a_props),
                        "kind": a_labels[0] if a_labels else "Node",
                        "community": a_props.get("leidenCommunity"),
                    }
                if b_key not in nodes:
                    nodes[b_key] = {
                        "id": b_key,
                        "label": _node_label(b_labels, b_props),
                        "kind": b_labels[0] if b_labels else "Node",
                        "community": b_props.get("leidenCommunity"),
                    }
                edges.append(
                    {
                        "source": a_key,
                        "target": b_key,
                        "rel": row["rel"],
                        "weight": float(r_props.get("weight") or 1.0),
                    }
                )
                if len(nodes) >= max_nodes:
                    break

            # Orphan papers/concepts with no edges yet still show up.
            if len(nodes) < max_nodes:
                extras = session.run(
                    """
                    MATCH (n)
                    WHERE any(l IN labels(n) WHERE l IN $labels)
                    RETURN labels(n) AS labels, properties(n) AS props
                    LIMIT $limit
                    """,
                    labels=labels,
                    limit=max_nodes,
                )
                for row in extras:
                    labs = list(row["labels"] or [])
                    props = dict(row["props"] or {})
                    key = _node_key(labs, props)
                    if key in nodes:
                        continue
                    nodes[key] = {
                        "id": key,
                        "label": _node_label(labs, props),
                        "kind": labs[0] if labs else "Node",
                        "community": props.get("leidenCommunity"),
                    }
                    if len(nodes) >= max_nodes:
                        break

            # Keep only edges whose endpoints survived the node cap.
            node_ids = set(nodes)
            edges = [
                e for e in edges if e["source"] in node_ids and e["target"] in node_ids
            ]

            return GraphSnapshot(
                nodes=list(nodes.values()),
                edges=edges,
                stats={
                    "node_counts": counts,
                    "shown_nodes": len(nodes),
                    "shown_edges": len(edges),
                    "communities": community_n,
                },
            )
    except Exception as exc:  # noqa: BLE001 - surface DB errors in the UI
        return GraphSnapshot(error=f"Neo4j unavailable: {exc}")
    finally:
        if owned:
            store.close()


def fetch_community_graph(
    *,
    store: Neo4jGraphStore | None = None,
    max_nodes: int = 80,
) -> GraphSnapshot:
    """Fetch Concept co-occurrence + community coloring (and paper membership)."""
    owned = store is None
    store = store or Neo4jGraphStore()
    try:
        with store.connect().session() as session:
            community_n = int(
                session.run("MATCH (c:Community) RETURN count(c) AS n").single()["n"]
            )
            rows = list(
                session.run(
                    """
                    MATCH (a:Concept)-[r:CO_OCCURS_WITH]->(b:Concept)
                    OPTIONAL MATCH (a)-[:BELONGS_TO]->(ca:Community)
                    OPTIONAL MATCH (b)-[:BELONGS_TO]->(cb:Community)
                    RETURN a.name AS a_name, a.id AS a_id,
                           coalesce(a.leidenCommunity, ca.leidenId) AS a_comm,
                           b.name AS b_name, b.id AS b_id,
                           coalesce(b.leidenCommunity, cb.leidenId) AS b_comm,
                           coalesce(r.weight, 1.0) AS weight
                    LIMIT $limit
                    """,
                    limit=max(50, max_nodes * 3),
                )
            )
            paper_rows = list(
                session.run(
                    """
                    MATCH (p:Paper)-[:BELONGS_TO]->(c:Community)
                    RETURN p.title AS title, p.id AS id, c.leidenId AS comm
                    LIMIT $limit
                    """,
                    limit=max_nodes,
                )
            )

            nodes: dict[str, dict[str, Any]] = {}
            edges: list[dict[str, Any]] = []
            for row in rows:
                a_key = f"Concept:{row['a_id'] or row['a_name']}"
                b_key = f"Concept:{row['b_id'] or row['b_name']}"
                nodes.setdefault(
                    a_key,
                    {
                        "id": a_key,
                        "label": str(row["a_name"] or "Concept")[:36],
                        "kind": "Concept",
                        "community": row["a_comm"],
                    },
                )
                nodes.setdefault(
                    b_key,
                    {
                        "id": b_key,
                        "label": str(row["b_name"] or "Concept")[:36],
                        "kind": "Concept",
                        "community": row["b_comm"],
                    },
                )
                edges.append(
                    {
                        "source": a_key,
                        "target": b_key,
                        "rel": "CO_OCCURS_WITH",
                        "weight": float(row["weight"] or 1.0),
                    }
                )
                if len(nodes) >= max_nodes:
                    break

            for row in paper_rows:
                if len(nodes) >= max_nodes:
                    break
                key = f"Paper:{row['id']}"
                title = str(row["title"] or "Paper")
                nodes[key] = {
                    "id": key,
                    "label": title if len(title) <= 42 else title[:39] + "…",
                    "kind": "Paper",
                    "community": row["comm"],
                }

            # Isolated concepts still in a community.
            if len(nodes) < max_nodes:
                for row in session.run(
                    """
                    MATCH (c:Concept)
                    OPTIONAL MATCH (c)-[:BELONGS_TO]->(comm:Community)
                    RETURN c.name AS name, c.id AS id,
                           coalesce(c.leidenCommunity, comm.leidenId) AS comm
                    LIMIT $limit
                    """,
                    limit=max_nodes,
                ):
                    key = f"Concept:{row['id'] or row['name']}"
                    if key in nodes:
                        continue
                    nodes[key] = {
                        "id": key,
                        "label": str(row["name"] or "Concept")[:36],
                        "kind": "Concept",
                        "community": row["comm"],
                    }
                    if len(nodes) >= max_nodes:
                        break

            node_ids = set(nodes)
            edges = [
                e for e in edges if e["source"] in node_ids and e["target"] in node_ids
            ]
            return GraphSnapshot(
                nodes=list(nodes.values()),
                edges=edges,
                stats={
                    "shown_nodes": len(nodes),
                    "shown_edges": len(edges),
                    "communities": community_n,
                },
            )
    except Exception as exc:  # noqa: BLE001
        return GraphSnapshot(error=f"Neo4j unavailable: {exc}")
    finally:
        if owned:
            store.close()


def _community_color(community: Any) -> str:
    if community is None:
        return "#94a3b8"
    try:
        idx = int(community) % len(COMMUNITY_PALETTE)
    except (TypeError, ValueError):
        idx = hash(str(community)) % len(COMMUNITY_PALETTE)
    return COMMUNITY_PALETTE[idx]


def build_plotly_figure(
    snapshot: GraphSnapshot,
    *,
    color_by_community: bool = False,
    title: str = "Knowledge graph",
):
    """Build an interactive Plotly network figure from a :class:`GraphSnapshot`."""
    import networkx as nx
    import plotly.graph_objects as go

    if snapshot.error:
        fig = go.Figure()
        fig.add_annotation(
            text=snapshot.error,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#b91c1c"},
        )
        fig.update_layout(
            title=title,
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=560,
            margin={"l": 20, "r": 20, "t": 48, "b": 20},
        )
        return fig

    if not snapshot.nodes:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "No graph nodes yet. Index papers with RAG sync enabled "
                "(Postgres + Neo4j running), then refresh."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(
            title=title,
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=560,
        )
        return fig

    graph = nx.Graph()
    for node in snapshot.nodes:
        graph.add_node(node["id"], **node)
    for edge in snapshot.edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            rel=edge.get("rel"),
            weight=edge.get("weight", 1.0),
        )

    if len(graph) == 1:
        pos = {list(graph.nodes)[0]: (0.0, 0.0)}
    else:
        pos = nx.spring_layout(graph, k=1.2 / max(len(graph) ** 0.5, 1), seed=42)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for src, dst in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    traces: list[Any] = [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1, "color": "#cbd5e1"},
            hoverinfo="none",
            showlegend=False,
        )
    ]

    # Group nodes for legend (by kind or community).
    groups: dict[str, list[dict[str, Any]]] = {}
    for node_id, data in graph.nodes(data=True):
        if color_by_community:
            comm = data.get("community")
            key = f"Community {comm}" if comm is not None else "Unassigned"
            color = _community_color(comm)
        else:
            key = str(data.get("kind") or "Node")
            color = LABEL_COLORS.get(key, "#64748b")
        groups.setdefault(key, []).append(
            {
                "x": pos[node_id][0],
                "y": pos[node_id][1],
                "text": data.get("label") or node_id,
                "color": color,
                "kind": data.get("kind"),
                "community": data.get("community"),
            }
        )

    for group_name, items in sorted(groups.items(), key=lambda kv: kv[0]):
        traces.append(
            go.Scatter(
                x=[i["x"] for i in items],
                y=[i["y"] for i in items],
                mode="markers+text",
                name=group_name,
                text=[i["text"] for i in items],
                textposition="top center",
                textfont={"size": 10},
                marker={
                    "size": [
                        18 if i["kind"] == "Paper" else 14 if i["kind"] == "Community" else 12
                        for i in items
                    ],
                    "color": items[0]["color"],
                    "line": {"width": 1, "color": "#0f172a"},
                },
                hovertext=[
                    f"{i['text']}<br>type={i['kind']}"
                    + (
                        f"<br>community={i['community']}"
                        if i.get("community") is not None
                        else ""
                    )
                    for i in items
                ],
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        height=620,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False},
        plot_bgcolor="#f8fafc",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def format_graph_stats(snapshot: GraphSnapshot) -> str:
    """Human-readable stats markdown for the Gradio side panel."""
    if snapshot.error:
        return f"**Graph error:** {snapshot.error}"
    stats = snapshot.stats or {}
    counts = stats.get("node_counts") or {}
    lines = [
        "### Graph stats",
        f"- Shown: **{stats.get('shown_nodes', len(snapshot.nodes))}** nodes, "
        f"**{stats.get('shown_edges', len(snapshot.edges))}** edges",
        f"- Communities: **{stats.get('communities', 0)}**",
    ]
    if counts:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- In Neo4j: {parts}")
    if stats.get("communities", 0) == 0:
        lines.append(
            "\n_No Leiden communities yet. Click **Run Leiden** to cluster concepts._"
        )
    return "\n".join(lines)


def render_graph_view(
    view: str = "Knowledge graph",
    max_nodes: int = 80,
    include_chunks: bool = False,
):
    """Return ``(plotly_figure, stats_markdown)`` for the Gradio Graph tab."""
    view_key = (view or "Knowledge graph").strip().lower()
    if view_key.startswith("community"):
        snapshot = fetch_community_graph(max_nodes=max_nodes)
        fig = build_plotly_figure(
            snapshot,
            color_by_community=True,
            title="Leiden communities (concepts + papers)",
        )
    else:
        snapshot = fetch_knowledge_graph(
            max_nodes=max_nodes,
            include_chunks=include_chunks,
        )
        fig = build_plotly_figure(
            snapshot,
            color_by_community=False,
            title="Academic knowledge graph",
        )
    return fig, format_graph_stats(snapshot)
