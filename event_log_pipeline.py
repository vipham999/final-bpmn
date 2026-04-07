"""
Process Mining + Graph2Vec + K-Means: build graphs from event logs (direct succession).
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def embed_graphs(
    graphs: List[nx.Graph],
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    try:
        from karateclub import Graph2Vec

        dim = min(32, max(16, len(graphs) * 2))
        model = Graph2Vec(
            dimensions=dim,
            workers=1,
            epochs=100,
            min_count=1,
            seed=int(random_state),
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

    embeddings, embed_mode = embed_graphs(graphs, random_state=random_state)
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


def graph_embeddings_dataframe(
    case_ids: List[str],
    embeddings: np.ndarray,
    clusters: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Ma trận embedding dạng bảng: mỗi hàng = một case, các cột dim_0 … dim_{d-1}.

    Tương đương ý nghĩa **graph embedding** trong notebook GNN (một vector cho cả đồ thị),
    nhưng đề tài dùng **Graph2Vec** thay vì GCN + global_mean_pool.
    """
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim != 2:
        raise ValueError("embeddings phai la ma tran 2D (n_case, dim).")
    n = emb.shape[0]
    if len(case_ids) != n:
        raise ValueError("So case_id khong khop so hang embedding.")
    cols = {f"dim_{j}": emb[:, j] for j in range(emb.shape[1])}
    df = pd.DataFrame({"case_id": [str(c) for c in case_ids], **cols})
    if clusters is not None:
        cl = list(clusters)
        if len(cl) == n:
            df.insert(1, "cluster", cl)
    return df


def graph_embeddings_from_pipeline_result(result: Dict[str, Any]) -> pd.DataFrame:
    """Gói tiện từ dict `run_pipeline` (case_ids, embeddings, clusters)."""
    return graph_embeddings_dataframe(
        result["case_ids"],
        result["embeddings"],
        clusters=result.get("clusters"),
    )


def graph_embeddings_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV UTF-8 BOM — mở tốt trên Excel."""
    return df.to_csv(index=False).encode("utf-8-sig")


def graph_embeddings_npz_bytes(case_ids: List[str], embeddings: np.ndarray) -> bytes:
    """NPZ nén: `case_ids` (object array), `embeddings` (float64 2D), cùng thứ tự hàng."""
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        case_ids=np.asarray(case_ids, dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float64),
    )
    buf.seek(0)
    return buf.read()


EMBEDDING_ARTIFACT_DIR = Path(__file__).resolve().parent / "outputs"
_EMB_CSV = "graph2vec_graph_embeddings.csv"
_EMB_NPZ = "graph2vec_graph_embeddings.npz"
_EMB_META = "graph2vec_run_meta.json"


def embedding_artifact_paths(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    root = Path(base_dir) if base_dir is not None else EMBEDDING_ARTIFACT_DIR
    return {
        "dir": root,
        "csv": root / _EMB_CSV,
        "npz": root / _EMB_NPZ,
        "meta": root / _EMB_META,
    }


def save_graph_embedding_artifacts(
    result: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Ghi đè CSV + NPZ + JSON meta trong thư mục `outputs/` (mặc định cạnh file này).
    Gọi sau mỗi lần huấn luyện thành công để xem lại bằng Excel / Python / app.
    """
    paths = embedding_artifact_paths(output_dir)
    root = paths["dir"]
    root.mkdir(parents=True, exist_ok=True)

    df = graph_embeddings_from_pipeline_result(result)
    paths["csv"].write_bytes(graph_embeddings_csv_bytes(df))

    paths["npz"].write_bytes(
        graph_embeddings_npz_bytes(result["case_ids"], result["embeddings"])
    )

    emb = np.asarray(result["embeddings"], dtype=float)
    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "embedding_mode": result["embedding_mode"],
        "n_clusters": int(result["n_clusters"]),
        "n_cases": len(result["case_ids"]),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "silhouette": result.get("silhouette"),
        "files": {
            "csv": _EMB_CSV,
            "npz": _EMB_NPZ,
            "meta": _EMB_META,
        },
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {k: str(v.resolve()) for k, v in paths.items() if k != "dir"} | {
        "dir": str(root.resolve())
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


def _bank20_catalog_meta() -> Dict[str, Dict[str, str]]:
    """Map process_id -> metadata from bank20 catalog (if available)."""
    p = bank_process_catalog_path()
    if not p.is_file():
        return {}
    try:
        cat = pd.read_csv(p)
    except Exception:
        return {}
    need = {"process_id", "process_code", "name_vn", "group_vn"}
    if not need.issubset(cat.columns):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for _, r in cat.iterrows():
        pid = str(r["process_id"]).strip()
        out[pid] = {
            "process_code": str(r["process_code"]).strip(),
            "name_vn": str(r["name_vn"]).strip(),
            "group_vn": str(r["group_vn"]).strip(),
        }
    return out


def variant_name_from_trace(
    trace: str,
    process_id: Optional[str] = None,
    bank_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Gan ma va ten bien the (tien Viet) dua tren mau trace.
    Dung cho demo + giai thich cum; khong thay the quy dinh nghiep vu.
    """
    if not trace or trace == TRACE_EMPTY:
        return ("BT-00", "Khong xac dinh")

    # Nhánh riêng cho bộ dữ liệu bank20: mã biến thể theo process_id (BK-xx).
    # Giúp tránh dồn toàn bộ về BT-99 khi trace không thuộc từ điển mẫu upload/demo.
    pid = str(process_id).strip() if process_id is not None else ""
    if pid.startswith("P") and pid[1:].isdigit():
        idx = int(pid[1:])
        meta = (bank_meta or {}).get(pid, {})
        pcode = meta.get("process_code", "")
        pname = meta.get("name_vn", "")
        pgroup = meta.get("group_vn", "")
        code = f"BK-{idx:02d}"
        title = f"Bank20 — {pid}"
        if pcode:
            title += f" ({pcode})"
        if pname:
            title += f": {pname}"
        if pgroup:
            title += f" [{pgroup}]"
        return (code, title)

    acts = [a.strip() for a in trace.split("→")]
    lower = [a.lower() for a in acts]
    compact = _norm_trace("→".join(acts))

    lo_chuan = _norm_trace(
        "LoanApplication→KYC→CreditScoring→CreditApproval→Disbursement"
    )
    if compact == lo_chuan:
        return ("BT-01", "Tín dụng chuẩn — đủ 5 bước (nộp hồ sơ → KYC → scoring → phê duyệt → giải ngân)")

    def kw(key: str) -> bool:
        return any(key in x.replace(" ", "") for x in lower)

    scoring_count = sum(1 for x in lower if "creditscoring" in x.replace(" ", ""))

    rules: List[Tuple[bool, Tuple[str, str]]] = [
        (kw("creditrejection"), ("BT-07", "Tín dụng — từ chối cấp tín dụng (CreditRejection)")),
        (
            kw("escalatecase") and not kw("creditscoring"),
            ("BT-09", "Tín dụng lệch — EscalateCase (không qua CreditScoring)"),
        ),
        (scoring_count >= 2, ("BT-05", "Tín dụng — lặp bước CreditScoring (làm lại / nhập lại)")),
        (
            kw("collateralcheck") and kw("legalreview"),
            ("BT-08", "Tín dụng hiếm — CollateralCheck + LegalReview"),
        ),
        (kw("legalreview"), ("BT-06", "Tín dụng — có LegalReview (pháp chế / hồ sơ phức tạp)")),
        (
            kw("collateralcheck") and kw("creditscoring"),
            ("BT-02", "Tín dụng phổ biến — thêm CollateralCheck (có tài sản đảm bảo)"),
        ),
        (
            kw("creditapproval") and not kw("creditscoring"),
            ("BT-03", "Tín dụng — bỏ CreditScoring (nhảy cóc / rút gọn quy trình)"),
        ),
        (
            len(acts) <= 3 and bool(lower) and lower[-1] == "withdrawnearly",
            ("BT-04", "Tín dụng — dừng sớm sau KYC (WithdrawnEarly)"),
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
        return "Gợi ý: từ chối cấp tín dụng / đóng hồ sơ — cần đối chiếu quy định tín dụng."
    if has_escalate and not has_scoring:
        return "Gợi ý: leo cấp xử lý trước khi có scoring — rất ít case, cần rà soát."
    if has_legal:
        return "Gợi ý: có LegalReview (pháp chế / hồ sơ phức tạp) — luồng thường dài hơn."
    if scoring_count >= 2:
        return "Gợi ý: lặp CreditScoring (làm lại điểm / bổ sung hồ sơ) — biến thể bất thường."
    if has_collateral and has_scoring:
        return f"Gợi ý: {n} bước, có thêm CollateralCheck (định giá/kiểm tra TSĐB)."
    if has_approval and not has_scoring:
        return "Gợi ý: quy trình ngắn — có thể thiếu CreditScoring; kiểm tra có đúng quy trình phê duyệt."
    if n <= 3 and norm and norm[-1] == "withdrawnearly":
        return "Gợi ý: khách/hồ sơ dừng sớm sau KYC (không tiếp tục giải ngân)."
    return f"Luồng nghiệp vụ {n} bước (theo thứ tự sự kiện trong hồ sơ)."


def dataframe_from_pipeline(
    result: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    cost_per_hour_vnd: float = 0.0,
) -> pd.DataFrame:
    rows = []
    traces: Optional[List[str]] = None
    cycle_by_case: Optional[pd.DataFrame] = None
    process_id_by_case: Dict[str, str] = {}
    bank_meta: Dict[str, Dict[str, str]] = {}
    if df is not None:
        traces = traces_aligned_with_result(df, result)
        cycle_by_case = case_cycle_metrics(df)
        if "process_id" in df.columns:
            first_pid = (
                df.sort_values("timestamp")
                .groupby("case_id", sort=False)["process_id"]
                .first()
                .astype(str)
            )
            process_id_by_case = {str(k): str(v) for k, v in first_pid.to_dict().items()}
            bank_meta = _bank20_catalog_meta()
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
            if process_id_by_case:
                row["process_id"] = process_id_by_case.get(str(cid), "")
            code, title = variant_name_from_trace(
                traces[i],
                process_id=process_id_by_case.get(str(cid), None),
                bank_meta=bank_meta,
            )
            row["ma_bien_the"] = code
            row["ten_bien_the"] = title
        rows.append(row)
    return pd.DataFrame(rows)


def _giai_thich_ma_bien_the(code: str, title: str) -> str:
    """
    Mô tả ngắn ý nghĩa mã hiển thị (BK-xx / BT-xx).
    BK-kk: ánh xạ process_id Pkk trong event log + catalog bank20.
    """
    code = (code or "").strip()
    title = (title or "").strip()
    if not code:
        return ""
    if code.startswith("BK-"):
        m = re.search(r"(P\d+)\s*\(([^)]+)\)", title)
        if m:
            pid, syscode = m.group(1), m.group(2)
            return (
                f"{code} = mã nội bộ gắn với **{pid}** (mã hệ thống `{syscode}` trong "
                f"`bank20_process_catalog.csv`). Quy ước: **{code} ↔ {pid}** (cùng số thứ tự)."
            )
        return (
            f"{code} = mã nội bộ theo cột **process_id** (P01…P20) trong event log; "
            "đối chiếu tên quy trình trong catalog Bank20."
        )
    if code.startswith("BT-"):
        return (
            f"{code} = mã biến thể theo **rule chuỗi activity** (bộ mẫu tín dụng/upload), "
            "khác hệ BK dùng cho bank20."
        )
    return f"Mã `{code}` — xem cột tên biến thể bên cạnh."


def cluster_variant_summary_table(out_df: pd.DataFrame) -> pd.DataFrame:
    """Mỗi dòng = 1 cụm K-Means: mã điển hình + giải thích + ghi chú có thể đọc được."""
    rows = []
    for c in sorted(out_df["cluster"].unique()):
        sub = out_df[out_df["cluster"] == c].copy()
        n = len(sub)
        if "ten_bien_the" not in sub.columns:
            rows.append(
                {
                    "cum_kmeans": int(c),
                    "ma_bien_the_dien_hinh": "",
                    "chi_tiet_ma_bien_the": "",
                    "ten_bien_the_dien_hinh": "",
                    "so_ho_so": n,
                    "ghi_chu": "",
                }
            )
            continue
        mode_title = sub["ten_bien_the"].mode()
        mode_code = sub["ma_bien_the"].mode()
        title = str(mode_title.iloc[0]) if len(mode_title) else ""
        code = str(mode_code.iloc[0]) if len(mode_code) else ""
        chi_tiet = _giai_thich_ma_bien_the(code, title)
        variants_in_cum = sub.groupby(["ma_bien_the", "ten_bien_the"]).size().reset_index(name="n")
        variants_in_cum = variants_in_cum.sort_values("n", ascending=False)
        if len(variants_in_cum) > 1:
            parts = [f"{r['ma_bien_the']} ({int(r['n'])} hồ sơ)" for _, r in variants_in_cum.iterrows()]
            ghi_chu = (
                f"Cụm trộn **{len(variants_in_cum)}** loại biến thể: " + "; ".join(parts) + ". "
                "Chi tiết từng hồ sơ: bảng **«Bảng chi tiết: từng hồ sơ — cụm — biến thể…»** ngay bên dưới."
            )
        else:
            ghi_chu = (
                "Cụm đồng nhất một loại biến thể (mọi hồ sơ cùng mã với dòng điển hình). "
                "Danh sách đầy đủ: bảng **«Bảng chi tiết: từng hồ sơ — cụm — biến thể…»** phía dưới."
            )
        rows.append(
            {
                "cum_kmeans": int(c),
                "ma_bien_the_dien_hinh": code,
                "chi_tiet_ma_bien_the": chi_tiet,
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


def bank20_csv_path() -> Path:
    """Event log: 20 quy trình ngân hàng (mỗi quy trình 2 case), cột process_id."""
    return Path(__file__).resolve().parent / "data" / "event_log_bank20.csv"


def bank_process_catalog_path() -> Path:
    """Catalog 20 quy trình: process_id, tên, nhóm, activity_sequence (|)."""
    return Path(__file__).resolve().parent / "data" / "bank20_process_catalog.csv"
