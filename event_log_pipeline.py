"""
Process Mining + Graph2Vec + K-Means: build graphs from event logs (direct succession).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

TRACE_EMPTY = "(rong)"

# Expected CSV columns (case-insensitive aliases accepted)
REQUIRED_ALIASES = {
    "case_id": ["case_id", "caseid", "case", "id"],
    "activity": ["activity", "task", "concept:name"],
    "timestamp": ["timestamp", "time", "datetime", "date"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in REQUIRED_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canon
                break
    out = df.rename(columns=rename)
    missing = set(REQUIRED_ALIASES) - set(out.columns)
    if missing:
        raise ValueError(f"Thieu cot: {missing}. Can co: case_id, activity, timestamp")
    return out


def load_event_log_csv(path_or_buffer: Any) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df = _normalize_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["case_id", "activity", "timestamp"])
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    return df


def _graph_to_int_labels(graph: nx.Graph) -> nx.Graph:
    nodes = list(graph.nodes())
    mapping = {n: i for i, n in enumerate(nodes)}
    return nx.relabel_nodes(graph, mapping)


def direct_succession_graph(case_df: pd.DataFrame) -> nx.Graph:
    """Alpha-style direct succession: sort by time, edge (a_i, a_{i+1})."""
    case_df = case_df.sort_values("timestamp")
    activities = case_df["activity"].tolist()
    graph = nx.Graph()
    if not activities:
        return graph
    for i in range(len(activities) - 1):
        graph.add_edge(activities[i], activities[i + 1])
    if len(activities) == 1:
        graph.add_node(activities[0])
    return _graph_to_int_labels(graph)


def build_graphs_per_case(df: pd.DataFrame) -> Tuple[List[str], List[nx.Graph]]:
    case_ids: List[str] = []
    graphs: List[nx.Graph] = []
    for cid, gdf in df.groupby("case_id", sort=True):
        case_ids.append(str(cid))
        graphs.append(direct_succession_graph(gdf))
    return case_ids, graphs


def embed_graphs(graphs: List[nx.Graph]) -> Tuple[np.ndarray, str]:
    try:
        from karateclub import Graph2Vec

        dim = min(32, max(16, len(graphs) * 2))
        model = Graph2Vec(
            dimensions=dim,
            workers=1,
            epochs=100,
            min_count=1,
        )
        model.fit(graphs)
        emb = model.get_embedding()
        return emb, "Graph2Vec (karateclub)"
    except Exception:
        emb = np.vstack([_structural_embedding(g) for g in graphs])
        return emb, "Fallback structural embedding"


def _structural_embedding(graph: nx.Graph) -> np.ndarray:
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    clustering = np.array(list(nx.clustering(graph).values()), dtype=float)
    return np.array(
        [
            graph.number_of_nodes(),
            graph.number_of_edges(),
            degrees.mean() if len(degrees) else 0.0,
            degrees.std() if len(degrees) else 0.0,
            np.max(degrees) if len(degrees) else 0.0,
            nx.density(graph),
            nx.number_connected_components(graph),
            clustering.mean() if len(clustering) else 0.0,
            nx.transitivity(graph),
            nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0.0,
        ],
        dtype=float,
    )


def run_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    case_ids, graphs = build_graphs_per_case(df)
    if len(graphs) < 2:
        raise ValueError("Can it nhat 2 case de phan cum.")

    embeddings, embed_mode = embed_graphs(graphs)
    n_clusters = max(2, min(n_clusters, len(graphs) - 1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    silhouette: Optional[float] = None
    if len(set(labels)) > 1 and len(graphs) > n_clusters:
        try:
            silhouette = float(
                silhouette_score(embeddings, labels, random_state=random_state)
            )
        except Exception:
            silhouette = None

    return {
        "case_ids": case_ids,
        "graphs": graphs,
        "embeddings": embeddings,
        "embedding_mode": embed_mode,
        "clusters": labels.tolist(),
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "silhouette": silhouette,
    }


def case_cycle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cycle time theo tung case: khoang thoi gian tu su kien dau den su kien cuoi (end-to-end).
    Mot su kien -> cycle_time = 0.
    """
    d = df.copy()
    d["case_id"] = d["case_id"].astype(str)
    agg = d.groupby("case_id", sort=True).agg(
        ts_dau=("timestamp", "min"),
        ts_cuoi=("timestamp", "max"),
    )
    delta = agg["ts_cuoi"] - agg["ts_dau"]
    # Gán trước khi reset_index: delta cùng index với agg (theo case_id).
    agg["cycle_time_s"] = delta.dt.total_seconds().astype(float)
    agg["cycle_time_gio"] = agg["cycle_time_s"] / 3600.0
    agg = agg.reset_index()
    return agg


def estimate_cost_vnd(cycle_time_gio: float, cost_per_hour_vnd: float) -> float:
    """Mo hinh don gian: chi phi van hanh ~ thoi gian x don gia / gio."""
    return float(cycle_time_gio) * float(cost_per_hour_vnd)


def cluster_cycle_cost_summary(out_df: pd.DataFrame) -> pd.DataFrame:
    """Tom tat theo cum: so case, cycle time trung binh / tong, tong chi phi (neu co)."""
    if "cluster" not in out_df.columns or "cycle_time_gio" not in out_df.columns:
        return pd.DataFrame()
    rows = []
    for c in sorted(out_df["cluster"].unique()):
        sub = out_df[out_df["cluster"] == c]
        row: Dict[str, Any] = {
            "cluster": int(c),
            "so_ho_so": len(sub),
            "cycle_tb_gio": float(sub["cycle_time_gio"].mean()),
            "cycle_tong_gio": float(sub["cycle_time_gio"].sum()),
        }
        if "chi_phi_vnd" in sub.columns:
            row["chi_phi_tong_vnd"] = float(sub["chi_phi_vnd"].sum())
            row["chi_phi_tb_vnd"] = float(sub["chi_phi_vnd"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def trace_string_for_case(df: pd.DataFrame, case_id: str) -> str:
    """Chuoi activity theo thoi gian, de nguoi doc hieu (khong dung so node)."""
    sub = df[df["case_id"] == str(case_id)].sort_values("timestamp")
    acts = sub["activity"].astype(str).tolist()
    return " → ".join(acts) if acts else TRACE_EMPTY


def traces_aligned_with_result(df: pd.DataFrame, result: Dict[str, Any]) -> List[str]:
    return [trace_string_for_case(df, cid) for cid in result["case_ids"]]


def _norm_trace(s: str) -> str:
    return s.replace(" ", "")


def variant_name_from_trace(trace: str) -> Tuple[str, str]:
    """
    Gan ma va ten bien the (tien Viet) dua tren mau trace.
    Dung cho demo + giai thich cum; khong thay the quy dinh nghiep vu.
    """
    if not trace or trace == TRACE_EMPTY:
        return ("BT-00", "Khong xac dinh")

    acts = [a.strip() for a in trace.split("→")]
    lower = [a.lower() for a in acts]
    compact = _norm_trace("→".join(acts))

    lo_chuan = _norm_trace(
        "LoanApplication→KYC→CreditScoring→CreditApproval→Disbursement"
    )
    if compact == lo_chuan:
        return ("BT-01", "LO chuẩn — đủ 5 bước (nộp hồ sơ → KYC → scoring → phê duyệt → giải ngân)")

    def kw(key: str) -> bool:
        return any(key in x.replace(" ", "") for x in lower)

    scoring_count = sum(1 for x in lower if "creditscoring" in x.replace(" ", ""))

    rules: List[Tuple[bool, Tuple[str, str]]] = [
        (kw("creditrejection"), ("BT-07", "LO — từ chối cấp tín dụng (CreditRejection)")),
        (
            kw("escalatecase") and not kw("creditscoring"),
            ("BT-09", "LO lệch — EscalateCase (không qua CreditScoring)"),
        ),
        (scoring_count >= 2, ("BT-05", "LO — lặp bước CreditScoring (làm lại / nhập lại)")),
        (
            kw("collateralcheck") and kw("legalreview"),
            ("BT-08", "LO hiếm — CollateralCheck + LegalReview"),
        ),
        (kw("legalreview"), ("BT-06", "LO — có LegalReview (pháp chế / hồ sơ phức tạp)")),
        (
            kw("collateralcheck") and kw("creditscoring"),
            ("BT-02", "LO phổ biến — thêm CollateralCheck (có tài sản đảm bảo)"),
        ),
        (
            kw("creditapproval") and not kw("creditscoring"),
            ("BT-03", "LO — bỏ CreditScoring (nhảy cóc / rút gọn quy trình)"),
        ),
        (
            len(acts) <= 3 and bool(lower) and lower[-1] == "withdrawnearly",
            ("BT-04", "LO — dừng sớm sau KYC (WithdrawnEarly)"),
        ),
    ]
    for ok, out in rules:
        if ok:
            return out
    return ("BT-99", "Khác — không khớp mẫu phân loại sẵn")


def describe_trace_vn(trace: str) -> str:
    """Mo ta ngan bang tieng Viet (goi y nghiep vu, khong thay the chuyen gia)."""
    if not trace or trace == TRACE_EMPTY:
        return "Khong co su kien."
    acts = [a.strip() for a in trace.split("→")]
    n = len(acts)
    lower = [a.lower() for a in acts]
    norm = [x.replace(" ", "") for x in lower]

    def has_sub(s: str) -> bool:
        return any(s in x for x in norm)

    has_collateral = has_sub("collateralcheck")
    has_scoring = has_sub("creditscoring")
    scoring_count = sum(1 for x in norm if "creditscoring" in x)
    has_approval = has_sub("creditapproval")
    has_reject = has_sub("creditrejection")
    has_legal = has_sub("legalreview")
    has_escalate = has_sub("escalatecase")
    if has_reject:
        return "LO: từ chối cấp tín dụng / đóng hồ sơ — cần đối chiếu quy định tín dụng."
    if has_escalate and not has_scoring:
        return "LO lệch: leo cấp xử lý trước khi có scoring — rất ít case, cần rà soát."
    if has_legal:
        return "LO: có LegalReview (pháp chế / hồ sơ phức tạp) — luồng thường dài hơn."
    if scoring_count >= 2:
        return "LO: lặp CreditScoring (làm lại điểm / bổ sung hồ sơ) — biến thể bất thường."
    if has_collateral and has_scoring:
        return f"LO: {n} bước, có thêm CollateralCheck (định giá/kiểm tra TSĐB)."
    if has_approval and not has_scoring:
        return "LO ngắn: có thể thiếu CreditScoring — kiểm tra có đúng quy trình phê duyệt."
    if n <= 3 and norm and norm[-1] == "withdrawnearly":
        return "LO: khách/hồ sơ dừng sớm sau KYC (không tiếp tục giải ngân)."
    return f"Luồng LO {n} bước (theo thứ tự sự kiện trong hồ sơ)."


def dataframe_from_pipeline(
    result: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    cost_per_hour_vnd: float = 0.0,
) -> pd.DataFrame:
    rows = []
    traces: Optional[List[str]] = None
    cycle_by_case: Optional[pd.DataFrame] = None
    if df is not None:
        traces = traces_aligned_with_result(df, result)
        cycle_by_case = case_cycle_metrics(df)
    for i, cid in enumerate(result["case_ids"]):
        g = result["graphs"][i]
        so_buoc = g.number_of_nodes()
        if traces and i < len(traces) and traces[i] not in (TRACE_EMPTY, ""):
            so_buoc = len([p for p in traces[i].split("→") if p.strip()])
        row: Dict[str, Any] = {
            "case_id": cid,
            "cluster": int(result["clusters"][i]),
            "so_buoc": so_buoc,
            "nodes_do_thi": g.number_of_nodes(),
            "edges_do_thi": g.number_of_edges(),
        }
        if cycle_by_case is not None:
            m = cycle_by_case[cycle_by_case["case_id"] == str(cid)]
            if not m.empty:
                row["ts_dau"] = m["ts_dau"].iloc[0]
                row["ts_cuoi"] = m["ts_cuoi"].iloc[0]
                row["cycle_time_s"] = float(m["cycle_time_s"].iloc[0])
                row["cycle_time_gio"] = float(m["cycle_time_gio"].iloc[0])
                cpe = float(cost_per_hour_vnd)
                row["chi_phi_vnd"] = estimate_cost_vnd(row["cycle_time_gio"], cpe) if cpe > 0 else 0.0
        if traces:
            row["chuoi_thao_tac"] = traces[i]
            row["goi_y"] = describe_trace_vn(traces[i])
            code, title = variant_name_from_trace(traces[i])
            row["ma_bien_the"] = code
            row["ten_bien_the"] = title
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_variant_summary_table(out_df: pd.DataFrame) -> pd.DataFrame:
    """Moi dong = 1 cum K-Means: ten bien the dien hinh + so case."""
    rows = []
    for c in sorted(out_df["cluster"].unique()):
        sub = out_df[out_df["cluster"] == c].copy()
        n = len(sub)
        if "ten_bien_the" not in sub.columns:
            rows.append(
                {
                    "cum_kmeans": int(c),
                    "ma_bien_the_dien_hinh": "",
                    "ten_bien_the_dien_hinh": "",
                    "so_ho_so": n,
                }
            )
            continue
        mode_title = sub["ten_bien_the"].mode()
        mode_code = sub["ma_bien_the"].mode()
        title = str(mode_title.iloc[0]) if len(mode_title) else ""
        code = str(mode_code.iloc[0]) if len(mode_code) else ""
        variants_in_cum = sub.groupby(["ma_bien_the", "ten_bien_the"]).size().reset_index(name="n")
        ghi_chu = ""
        if len(variants_in_cum) > 1:
            ghi_chu = "Trong cum co nhieu loai bien the (xem bang chi tiet)."
        rows.append(
            {
                "cum_kmeans": int(c),
                "ma_bien_the_dien_hinh": code,
                "ten_bien_the_dien_hinh": title,
                "so_ho_so": n,
                "ghi_chu": ghi_chu,
            }
        )
    return pd.DataFrame(rows)


def cluster_story_markdown(
    df: pd.DataFrame, result: Dict[str, Any], cost_per_hour_vnd: float = 0.0
) -> str:
    """Tom tat tung cum bang loi: gan voi ten bien the."""
    out_df = dataframe_from_pipeline(result, df, cost_per_hour_vnd=cost_per_hour_vnd)
    lines: List[str] = ["### Tóm tắt từng cụm (theo biến thể nhận diện)\n"]
    summary = cluster_variant_summary_table(out_df)
    for _, row in summary.iterrows():
        c = int(row["cum_kmeans"])
        code = row.get("ma_bien_the_dien_hinh", "")
        title = row.get("ten_bien_the_dien_hinh", "")
        n = int(row["so_ho_so"])
        note = row.get("ghi_chu", "")
        lines.append(f"**Cụm {c}** — `{code}` *{title}* — **{n}** hồ sơ.")
        if note:
            lines.append(f"- {note}")
        sub = out_df[out_df["cluster"] == c]
        sample = sub.iloc[0]
        trace = sample.get("chuoi_thao_tac", "")
        if trace:
            lines.append(f"- Ví dụ trace: `{sample['case_id']}` → {trace}")
            lines.append(f"- Gợi ý: {sample.get('goi_y', '')}")
        lines.append("")
    return "\n".join(lines)


def demo_csv_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "event_log_demo.csv"
