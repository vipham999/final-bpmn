"""
Cosine similarity trên graph embeddings: truy vấn trùng lặp & cặp gần trùng.
Tách file để app.py không phụ thuộc vào việc event_log_pipeline.py có đủ hàm hay không.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def rank_similarity_to_query(
    case_ids: List[str],
    embeddings: np.ndarray,
    query_case_id: str,
) -> pd.DataFrame:
    """
    So sánh embedding của một case (truy vấn) với toàn bộ case trong kho.
    Độ đo: cosine similarity (1 = cùng hướng vector, gần trùng cấu trúc theo Graph2Vec).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings phai la ma tran 2 chieu (n_cases, dim).")
    q = str(query_case_id)
    if q not in case_ids:
        raise ValueError(f"Khong tim thay case_id: {q}")

    qi = case_ids.index(q)
    vec_q = embeddings[qi : qi + 1]
    sims = cosine_similarity(vec_q, embeddings)[0]

    out = pd.DataFrame(
        {
            "case_id": case_ids,
            "cosine_similarity": sims.astype(float),
            "la_chinh_no": [str(c) == q for c in case_ids],
        }
    )
    out = out.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
    out["tuong_dong_pct"] = (out["cosine_similarity"] * 100.0).round(2)
    return out


def pairs_exceeding_similarity(
    case_ids: List[str],
    embeddings: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Liệt kê các cặp case khác nhau có cosine similarity >= threshold (ma trận đối xứng, chỉ nửa trên).
    """
    if len(case_ids) < 2:
        return pd.DataFrame(columns=["case_a", "case_b", "cosine_similarity", "tuong_dong_pct"])
    mat = cosine_similarity(embeddings)
    rows: List[Dict[str, Any]] = []
    n = len(case_ids)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(mat[i, j])
            if s >= threshold:
                rows.append(
                    {
                        "case_a": case_ids[i],
                        "case_b": case_ids[j],
                        "cosine_similarity": s,
                        "tuong_dong_pct": round(s * 100.0, 2),
                    }
                )
    cols = ["case_a", "case_b", "cosine_similarity", "tuong_dong_pct"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)


def exact_trace_duplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Các cặp case có **chuỗi activity y hệt** (cùng thứ tự theo timestamp).
    Dùng đối chiếu với Graph2Vec: Doc2Vec có thể gán vector khác nhau cho hai đồ thị
    dù WL-features trùng, nên cosine không luôn ~1 dù quy trình giống hệt.
    """
    need = {"case_id", "activity", "timestamp"}
    if not need.issubset(df.columns):
        return pd.DataFrame(
            columns=["case_a", "case_b", "so_buoc", "trace_preview"]
        )
    d = df.copy()
    d["case_id"] = d["case_id"].astype(str)
    seq_by_case: Dict[str, Tuple[str, ...]] = {}
    for cid, g in d.groupby("case_id", sort=False):
        g2 = g.sort_values("timestamp")
        seq_by_case[str(cid)] = tuple(g2["activity"].astype(str).tolist())

    buckets: Dict[Tuple[str, ...], List[str]] = {}
    for cid, seq in seq_by_case.items():
        buckets.setdefault(seq, []).append(cid)

    rows: List[Dict[str, Any]] = []
    for seq, cases in buckets.items():
        if len(cases) < 2:
            continue
        cases_sorted = sorted(cases)
        for a, b in combinations(cases_sorted, 2):
            prev = " → ".join(seq[:4])
            if len(seq) > 4:
                prev += " → …"
            rows.append(
                {
                    "case_a": a,
                    "case_b": b,
                    "so_buoc": len(seq),
                    "trace_preview": prev,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["case_a", "case_b", "so_buoc", "trace_preview"])
    return pd.DataFrame(rows).sort_values(["so_buoc", "case_a", "case_b"]).reset_index(drop=True)


def duplicate_process_pairs_from_catalog(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Các **cặp quy trình** (process_id khác nhau) có **cùng định nghĩa luồng**
    (`activity_sequence` giống hệt). Đây là tiêu chí trùng ở mức *nghiệp vụ / BPM*.
    """
    need = {"process_id", "activity_sequence", "name_vn"}
    if not need.issubset(catalog_df.columns):
        return pd.DataFrame(
            columns=[
                "process_a",
                "process_b",
                "name_a",
                "name_b",
                "so_buoc",
                "trace_preview",
            ]
        )

    buckets: Dict[Tuple[str, ...], List[str]] = {}
    names: Dict[str, str] = {}
    for _, row in catalog_df.iterrows():
        pid = str(row["process_id"]).strip()
        names[pid] = str(row["name_vn"])
        seq = tuple(x.strip() for x in str(row["activity_sequence"]).split("|") if x.strip())
        buckets.setdefault(seq, []).append(pid)

    rows: List[Dict[str, Any]] = []
    for seq, pids in buckets.items():
        if len(pids) < 2:
            continue
        pids_u = sorted(set(pids))
        for a, b in combinations(pids_u, 2):
            prev = " → ".join(seq[:5])
            if len(seq) > 5:
                prev += " → …"
            rows.append(
                {
                    "process_a": a,
                    "process_b": b,
                    "name_a": names.get(a, ""),
                    "name_b": names.get(b, ""),
                    "so_buoc": len(seq),
                    "trace_preview": prev,
                }
            )
    cols = [
        "process_a",
        "process_b",
        "name_a",
        "name_b",
        "so_buoc",
        "trace_preview",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["process_a", "process_b"]).reset_index(drop=True)


def process_groups_identical_flow(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Mỗi hàng = một nhóm process_id chia sẻ cùng một activity_sequence (|N| >= 2)."""
    need = {"process_id", "activity_sequence", "name_vn"}
    if not need.issubset(catalog_df.columns):
        return pd.DataFrame(columns=["nhom_stt", "process_ids", "ten_quy_trinh", "so_buoc", "trace_preview"])

    buckets: Dict[Tuple[str, ...], List[str]] = {}
    names: Dict[str, str] = {}
    for _, row in catalog_df.iterrows():
        pid = str(row["process_id"]).strip()
        names[pid] = str(row["name_vn"])
        seq = tuple(x.strip() for x in str(row["activity_sequence"]).split("|") if x.strip())
        buckets.setdefault(seq, []).append(pid)

    out_rows: List[Dict[str, Any]] = []
    gno = 0
    for seq, pids in sorted(buckets.items(), key=lambda x: min(x[1])):
        pids_u = sorted(set(pids))
        if len(pids_u) < 2:
            continue
        gno += 1
        prev = " → ".join(seq[:5])
        if len(seq) > 5:
            prev += " → …"
        out_rows.append(
            {
                "nhom_stt": gno,
                "process_ids": ", ".join(pids_u),
                "ten_quy_trinh": " | ".join(f"{p}: {names.get(p, '')}" for p in pids_u),
                "so_buoc": len(seq),
                "trace_preview": prev,
            }
        )
    if not out_rows:
        return pd.DataFrame(columns=["nhom_stt", "process_ids", "ten_quy_trinh", "so_buoc", "trace_preview"])
    return pd.DataFrame(out_rows)
