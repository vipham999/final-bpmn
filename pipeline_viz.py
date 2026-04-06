"""Trực quan hoá: Plotly (PCA, heatmap), NetworkX, Pyvis (succession)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def clusters_for_case_order(
    case_ids: list[str], out_df: pd.DataFrame | None
) -> np.ndarray | None:
    if out_df is None or "cluster" not in out_df.columns:
        return None
    cmap = out_df[["case_id", "cluster"]].drop_duplicates()
    m = {str(r.case_id): int(r.cluster) for r in cmap.itertuples()}
    return np.array([m.get(str(c), 0) for c in case_ids])


def build_succession_digraph(df: pd.DataFrame, case_id: str) -> nx.DiGraph:
    """Đồ thị có hướng: cạnh activity_i → activity_{i+1} theo timestamp."""
    sub = df[df["case_id"].astype(str) == str(case_id)].sort_values("timestamp")
    G = nx.DiGraph()
    if sub.empty or "activity" not in sub.columns:
        return G
    acts = sub["activity"].astype(str).tolist()
    if len(acts) == 1:
        G.add_node(acts[0])
    for i in range(len(acts) - 1):
        G.add_edge(acts[i], acts[i + 1])
    return G


def succession_pyvis_html(df: pd.DataFrame, case_id: str) -> str:
    """HTML đầy đủ cho Pyvis: kéo nút, zoom, vật lý — nhúng qua streamlit.components.v1.html."""
    from pyvis.network import Network

    G = build_succession_digraph(df, case_id)
    if G.number_of_nodes() == 0:
        return (
            "<html><body style='font-family:sans-serif;padding:1rem;'>"
            f"<p>Không có dữ liệu activity cho hồ sơ <b>{case_id}</b>.</p></body></html>"
        )

    net = Network(
        height="460px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="remote",
    )
    net.barnes_hut(gravity=-3500, central_gravity=0.35, spring_length=140)

    for n in G.nodes():
        net.add_node(
            n,
            label=str(n),
            title=str(n),
            shape="box",
            color="#e8daef",
            border="#5b2c6f",
        )
    for u, v in G.edges():
        net.add_edge(u, v, arrows="to")

    return net.generate_html()


def figure_embedding_pca_scatter(
    embeddings: np.ndarray,
    case_ids: list[str],
    clusters: np.ndarray | None,
) -> go.Figure:
    X = np.asarray(embeddings, dtype=float)
    n = X.shape[0]
    if n < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Cần ít nhất 2 hồ sơ để vẽ PCA.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template="plotly_white", height=320)
        return fig

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(X)
    # Plotly marker.colorscale chỉ nhận bảng liên tục (Viridis, Turbo…), không có "Set2".
    if clusters is not None and len(clusters) == n:
        uniq = np.sort(np.unique(clusters))
        palette = (px.colors.qualitative.Plotly + px.colors.qualitative.Dark24) * 2
        fig = go.Figure()
        case_arr = np.asarray(case_ids, dtype=object)
        for i, k in enumerate(uniq):
            mask = clusters == k
            fig.add_trace(
                go.Scatter(
                    x=xy[mask, 0],
                    y=xy[mask, 1],
                    mode="markers",
                    marker=dict(size=11, color=palette[i % len(palette)]),
                    text=[
                        f"{cid}<br>Cụm {int(k)}"
                        for cid in case_arr[mask].tolist()
                    ],
                    name=f"Cụm {int(k)}",
                    hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
                )
            )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xy[:, 0],
                    y=xy[:, 1],
                    mode="markers",
                    marker=dict(size=11, color="#636EFA"),
                    text=[str(cid) for cid in case_ids],
                    hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
                )
            ],
        )
    var = pca.explained_variance_ratio_
    fig.update_layout(
        title="PCA 2D trên embedding Graph2Vec (màu = cụm K-Means)",
        xaxis_title=f"PC1 ({var[0] * 100:.1f}% phương sai)",
        yaxis_title=f"PC2 ({var[1] * 100:.1f}% phương sai)",
        template="plotly_white",
        height=480,
        margin=dict(l=48, r=48, t=56, b=48),
    )
    return fig


def figure_cosine_heatmap(
    embeddings: np.ndarray,
    case_ids: list[str],
    clusters: np.ndarray | None = None,
    max_cases: int = 100,
    cosine_dampen: float = 1.0,
) -> tuple[go.Figure, str | None]:
    """Trả về (figure, cảnh báo nếu đã cắt bớt case)."""
    n = len(case_ids)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=320)
        return fig, None

    warn = None
    if n > max_cases:
        warn = f"Chỉ hiển thị **{max_cases}** / {n} hồ sơ để ma trận đọc được (cosine tính trên tập con đầu tiên)."
        idx = np.arange(max_cases)
        case_ids = [case_ids[i] for i in idx]
        emb = np.asarray(embeddings, dtype=float)[idx]
        if clusters is not None:
            clusters = clusters[idx]
    else:
        emb = np.asarray(embeddings, dtype=float)

    sim = cosine_similarity(emb)
    if cosine_dampen < 1.0:
        sim = sim.astype(float) * float(cosine_dampen)
        np.fill_diagonal(sim, 1.0)
        sim = np.clip(sim, -1.0, 1.0)
    case_ids_str = [str(c) for c in case_ids]

    if clusters is not None and len(clusters) == len(case_ids_str):
        order = sorted(
            range(len(case_ids_str)),
            key=lambda i: (int(clusters[i]), case_ids_str[i]),
        )
    else:
        order = list(range(len(case_ids_str)))

    ordered_ids = [case_ids_str[i] for i in order]
    sim_ord = sim[np.ix_(order, order)]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=sim_ord,
                x=ordered_ids,
                y=ordered_ids,
                colorscale="Blues",
                zmin=0.0,
                zmax=1.0,
                hovertemplate="Hồ sơ A: %{y}<br>Hồ sơ B: %{x}<br>Cosine: %{z:.3f}<extra></extra>",
            )
        ]
    )
    h = min(900, max(420, 10 * len(ordered_ids)))
    fig.update_layout(
        title="Ma trận cosine giữa các embedding (hàng/cột sắp theo cụm nếu có)",
        template="plotly_white",
        height=h,
        margin=dict(l=120, r=40, t=56, b=120),
        xaxis=dict(side="bottom", tickangle=-55),
        yaxis=dict(autorange="reversed"),
    )
    return fig, warn


def figure_succession_digraph(df: pd.DataFrame, case_id: str) -> go.Figure:
    """Plotly: đồ thị succession (tùy chọn; app hiện dùng Pyvis)."""
    G = build_succession_digraph(df, case_id)
    fig = go.Figure()
    if G.number_of_nodes() == 0:
        fig.add_annotation(
            text="Không có dữ liệu activity cho hồ sơ này.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template="plotly_white", height=360, title=f"Case {case_id}")
        return fig

    k = 2.0 / max(G.number_of_nodes(), 1) ** 0.5
    pos = nx.spring_layout(G, seed=42, k=k)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="Kề cận",
    )

    nx_ = [pos[n][0] for n in G.nodes()]
    ny_ = [pos[n][1] for n in G.nodes()]
    labels = list(G.nodes())

    node_trace = go.Scatter(
        x=nx_,
        y=ny_,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=22, color="#AB63FA", line=dict(width=1, color="#333")),
        hovertemplate="%{text}<extra></extra>",
        name="Hoạt động",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Đồ thị succession (trực tiếp) — hồ sơ {case_id}",
        showlegend=False,
        template="plotly_white",
        height=520,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=56, b=20),
    )
    return fig
