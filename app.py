import io
import json
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SO_HO_SO_LABEL = "Số hồ sơ"
_ROOT = Path(__file__).resolve().parent
_DIAGRAM_BPMN = _ROOT / "docs" / "diagrams" / "bpmn_bank20_overview.svg"
_DIAGRAM_PN = _ROOT / "docs" / "diagrams" / "petrinet_bank20_template.svg"
_BANK20_CSV = _ROOT / "data" / "event_log_bank20.csv"


def _render_linear_bpmn_svg(activity_sequence: list[str], title: str) -> str:
    """Render BPMN happy-path SVG from a linear activity sequence."""
    n = len(activity_sequence)
    box_w = 132
    box_h = 54
    gap = 26
    margin_x = 26
    start_r = 14
    top_y = 92
    content_w = margin_x * 2 + 80 + (n * box_w) + ((n + 1) * gap) + 80
    width = max(920, content_w)
    height = 240

    x = margin_x + start_r
    start_cx = x
    start_cy = top_y + box_h // 2
    x += start_r + gap

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#222"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f9f9f9"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="22" font-weight="600" fill="#222">{escape(title)}</text>',
        f'<circle cx="{start_cx}" cy="{start_cy}" r="{start_r}" fill="white" stroke="#222" stroke-width="3"/>',
        f'<line x1="{start_cx + start_r}" y1="{start_cy}" x2="{x - 8}" y2="{start_cy}" stroke="#222" stroke-width="2.5" marker-end="url(#arrow)"/>',
    ]

    for idx, act in enumerate(activity_sequence):
        rx = x
        ry = top_y
        parts.append(
            f'<rect x="{rx}" y="{ry}" width="{box_w}" height="{box_h}" rx="8" fill="#dbe9f2" stroke="#2f81c1" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{rx + box_w/2:.1f}" y="{ry + 33}" text-anchor="middle" font-size="13" fill="#1f2937">{escape(act)}</text>'
        )
        x = rx + box_w + gap
        if idx < n - 1:
            parts.append(
                f'<line x1="{rx + box_w}" y1="{start_cy}" x2="{x - 8}" y2="{start_cy}" stroke="#222" stroke-width="2.5" marker-end="url(#arrow)"/>'
            )

    end_cx = x + start_r
    parts.append(
        f'<line x1="{x - gap}" y1="{start_cy}" x2="{end_cx - start_r - 8}" y2="{start_cy}" stroke="#222" stroke-width="2.5" marker-end="url(#arrow)"/>'
    )
    parts.append(f'<circle cx="{end_cx}" cy="{start_cy}" r="{start_r}" fill="white" stroke="#222" stroke-width="3"/>')
    parts.append(
        f'<circle cx="{end_cx}" cy="{start_cy}" r="{start_r - 5}" fill="none" stroke="#222" stroke-width="2"/>'
    )
    parts.append(f'<text x="{start_cx - 4}" y="{height - 22}" font-size="14" fill="#444">Start</text>')
    parts.append(f'<text x="{end_cx - 12}" y="{height - 22}" font-size="14" fill="#444">End</text>')
    parts.append("</svg>")
    return "".join(parts)


def _diagram_process_options() -> pd.DataFrame:
    cat_df = pd.read_csv(bank_process_catalog_path()).copy()
    cat_df["activity_sequence_list"] = (
        cat_df["activity_sequence"].astype(str).str.split("|").apply(lambda s: [x.strip() for x in s if x.strip()])
    )
    return cat_df


def render_diagrams_web_tab() -> None:
    """Tab sơ đồ: render BPMN động theo dataset/process_id, kèm Petri net mẫu."""
    st.subheader("BPMN 2.0 — render động theo quy trình")
    st.caption("Nguồn BPMN động: `data/bank20_process_catalog.csv`.")
    try:
        options_df = _diagram_process_options()
    except FileNotFoundError as exc:
        st.warning(f"Không tìm thấy dữ liệu để vẽ BPMN: {exc}")
        return
    except ValueError as exc:
        st.warning(str(exc))
        return

    options_df = options_df.sort_values("process_id").reset_index(drop=True)
    pick_label = options_df.apply(
        lambda r: f"{r['process_id']} — {r.get('process_code', '')} — {r.get('name_vn', '')}",
        axis=1,
    ).tolist()
    selected_label = st.selectbox(
        "Chọn process_id",
        options=pick_label,
        index=0,
        key="diagram_process_id",
    )
    selected_idx = pick_label.index(selected_label)
    selected = options_df.iloc[selected_idx]
    acts = selected["activity_sequence_list"]
    if not isinstance(acts, list) or not acts:
        st.warning("Không có chuỗi activity để vẽ BPMN.")
        return

    title = f"BPMN 2.0 - {selected['process_id']} ({selected.get('process_code', '')})"
    st.caption("Start → " + " → ".join(acts) + " → End")
    components.html(_render_linear_bpmn_svg(acts, title=title), height=260, scrolling=True)

    with st.expander("Sơ đồ BPMN/Petri mẫu cố định (tài liệu gốc)", expanded=False):
        if _DIAGRAM_BPMN.is_file():
            st.image(str(_DIAGRAM_BPMN), width="stretch")
        else:
            st.warning(f"Không tìm thấy file: {_DIAGRAM_BPMN}")

    st.subheader("Petri net — Workflow net (place • → transition ▭ → …)")
    st.caption(
        "Ô tròn = place (trạng thái), hình chữ nhật = transition (hoạt động). Token ở p₀ = một case đang xử lý."
    )
    if _DIAGRAM_PN.is_file():
        st.image(str(_DIAGRAM_PN), width="stretch")
    else:
        st.warning(f"Không tìm thấy file: {_DIAGRAM_PN}")

    st.info(
        "Bạn có thể xem BPMN động theo từng process ở phía trên; hai file tĩnh gốc vẫn nằm ở "
        "`docs/diagrams/bpmn_bank20_overview.svg`, `docs/diagrams/petrinet_bank20_template.svg`."
    )


def cluster_count_bar_chart(counts_sorted: pd.Series) -> None:
    """
    Vẽ cột theo đúng thứ tự trong counts_sorted (trái → phải).
    Dùng Altair vì st.bar_chart hay sắp lại theo tên cụm 0,1,2…
    """
    labels = [f"Cụm {c}" for c in counts_sorted.index]
    frame = pd.DataFrame({"cum_label": labels, "n": counts_sorted.values.astype(int)})
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("cum_label:N", sort=labels, title="Cụm"),
            y=alt.Y("n:Q", title=SO_HO_SO_LABEL),
            tooltip=[
                alt.Tooltip("cum_label:N", title="Cụm"),
                alt.Tooltip("n:Q", title=SO_HO_SO_LABEL),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)

from duplicate_search import (
    duplicate_process_pairs_from_catalog,
    exact_trace_duplicate_pairs,
    pairs_exceeding_similarity,
    process_groups_identical_flow,
    rank_similarity_to_query,
)
from pipeline_viz import (
    clusters_for_case_order,
    figure_cosine_heatmap,
    figure_embedding_pca_scatter,
    succession_pyvis_html,
)
from event_log_pipeline import (
    cluster_cycle_cost_summary,
    cluster_story_markdown,
    cluster_variant_summary_table,
    dataframe_from_pipeline,
    bank_process_catalog_path,
    embedding_artifact_paths,
    graph_embeddings_csv_bytes,
    graph_embeddings_from_pipeline_result,
    graph_embeddings_npz_bytes,
    load_event_log_csv,
    run_pipeline,
    save_graph_embedding_artifacts,
)

def render_event_log_thesis_tab() -> None:
    st.subheader("Bank20 — Event log → đồ thị → Graph2Vec → K-Means")
    _art = embedding_artifact_paths()
    if _art["csv"].is_file():
        with st.expander(
            "📂 Graph embedding đã lưu trên đĩa (`outputs/`) — xem lại không cần train",
            expanded=False,
        ):
            st.caption(
                "Mỗi lần **Huấn luyện** thành công, app ghi đè 3 file trong thư mục `outputs/` của project."
            )
            st.code(str(_art["csv"].resolve()), language=None)
            if _art["meta"].is_file():
                try:
                    st.json(json.loads(_art["meta"].read_text(encoding="utf-8")))
                except Exception:
                    pass
            try:
                st.dataframe(
                    pd.read_csv(_art["csv"]),
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                )
            except Exception as exc:
                st.warning(f"Không đọc được CSV: {exc}")

    st.markdown(
        """
**Kịch bản demo:** dùng bộ **20 quy trình ngân hàng** (nhiều hồ sơ/case, nhiều biến thể / trùng cấu trúc). Mỗi `case_id` là một hồ sơ; `activity` là bước nghiệp vụ.

1. **Trích đồ thị**: với mỗi `case_id`, sắp xếp theo `timestamp`, nối cạnh giữa activity liền kề (succession trực tiếp).
2. **Model 1 – Graph2Vec**: `fit` trên đồ thị của từng hồ sơ → vector embedding (graph embedding).
3. **Model 2 – K-Means**: phân cụm các vector → gom các **biến thể quy trình** có cấu trúc tương đồng.
4. **Lọc trùng lặp (tùy chọn sau khi huấn luyện)**: **cosine similarity** giữa vector của một hồ sơ truy vấn và toàn kho; ngưỡng (ví dụ ≥ 90%) để cảnh báo **cấu trúc quy trình gần trùng**.
"""
    )
    with st.expander("Bạn sẽ đọc kết quả thế nào? (không chỉ là con số)", expanded=False):
        st.markdown(
            """
- **Cột “Cụm” (cluster)**: máy gom các **hồ sơ (case)** có **cách đi quy trình giống nhau** vào một nhóm. Cùng cụm ≠ cùng đúng/sai nghiệp vụ, mà là **cùng kiểu luồng**.
- **Chuỗi thao tác**: đúng thứ tự **activity** trong từng hồ sơ — dễ hiểu nhất; “gợi ý” là mô tả nhanh theo luồng nghiệp vụ.
- **Silhouette** (nếu có): đo độ “tách bạch” giữa các cụm (càng gần **1** thường càng rõ ràng). Chỉ mang tính **tham khảo**, không phải kết luận cuối.
- **Node/edge đồ thị**: dùng nội bộ để so khớp cấu trúc; có thể xem trong phần kỹ thuật.
"""
        )

    demo_mode = st.radio(
        "Nguồn dữ liệu",
        options=["bank20", "upload"],
        format_func=lambda x: {
            "bank20": "Ngân hàng — **20 quy trình** — `event_log_bank20.csv` + `bank20_process_catalog.csv`",
            "upload": "Upload CSV (case_id, activity, timestamp)",
        }[x],
        index=0,
        horizontal=True,
        key="demo_data_source",
    )
    if demo_mode == "bank20":
        st.caption(
            "Mỗi **quy trình** (P01…P20) có một **chuỗi activity** cố định trong catalog; event log có thêm cột "
            "**`process_id`** để đối chiếu. "
            "**Trùng quy trình** = hai mã P khác nhau nhưng cùng `activity_sequence` (ví dụ P01↔P14, P03↔P12, P06↔P13). "
            "Activity vocabulary: `data/bank20_activity_catalog.csv`. Sinh lại dữ liệu: `python scripts/generate_bank20_processes.py`."
        )
        try:
            cat_df = pd.read_csv(bank_process_catalog_path())
            with st.expander(
                "📋 20 quy trình — danh mục & quy trình nào **trùng luồng** với nhau",
                expanded=True,
            ):
                st.dataframe(
                    cat_df.rename(
                        columns={
                            "process_id": "Mã QTR",
                            "process_code": "Mã hệ thống",
                            "name_vn": "Tên quy trình",
                            "group_vn": "Nhóm",
                            "activity_sequence": "Chuỗi activity (|)",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )
                grp = process_groups_identical_flow(cat_df)
                st.markdown("**Nhóm quy trình có luồng giống hệt** (cần hợp nhất tài liệu / BPM nếu trùng).")
                st.dataframe(
                    grp.rename(
                        columns={
                            "nhom_stt": "Nhóm #",
                            "process_ids": "Mã quy trình",
                            "ten_quy_trinh": "Tên (theo mã)",
                            "so_buoc": "Số bước",
                            "trace_preview": "Đầu luồng",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                prs = duplicate_process_pairs_from_catalog(cat_df)
                st.markdown("**Cặp quy trình trùng** (từng cặp P↔P).")
                st.dataframe(
                    prs.rename(
                        columns={
                            "process_a": "QTR A",
                            "process_b": "QTR B",
                            "name_a": "Tên A",
                            "name_b": "Tên B",
                            "so_buoc": "Số bước",
                            "trace_preview": "Đầu luồng",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        except FileNotFoundError:
            st.warning(
                "Không thấy `data/bank20_process_catalog.csv`. Chạy: `python scripts/generate_bank20_processes.py`"
            )
    uploaded = None
    if demo_mode == "upload":
        uploaded = st.file_uploader("Upload CSV (cột: case_id, activity, timestamp)", type=["csv"])

    n_clusters = st.slider("Số cụm K (K-Means)", 2, 8, 3, 1, key="kmeans_k")
    seed = st.number_input("random_state", value=42, min_value=0, step=1)
    cost_per_gio = st.number_input(
        "Đơn giá vận hành ước lượng (VNĐ / giờ) — dùng tính chi phí",
        min_value=0.0,
        value=500_000.0,
        step=50_000.0,
        format="%.0f",
        help="Chi phí = (thời gian chu kỳ tính bằng giờ) × đơn giá. Mô hình minh họa; thay bằng số liệu đơn vị bạn.",
    )
    st.caption(
        "**Chu kỳ (cycle time)** mỗi hồ sơ = thời gian từ **sự kiện đầu tiên** đến **sự kiện cuối** theo `timestamp`."
    )

    if st.button("Huấn luyện Graph2Vec + K-Means", type="primary", key="train_pipeline"):
        try:
            if demo_mode == "upload":
                if uploaded is None:
                    st.warning("Hãy upload file CSV.")
                    return
                df = load_event_log_csv(io.BytesIO(uploaded.read()))
            else:
                df = load_event_log_csv(_BANK20_CSV)

            st.write("**Xem trước log** (10 dòng đầu)")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            st.caption(
                f"Tổng **{len(df)}** dòng, **{df['case_id'].nunique()}** case."
            )

            result = run_pipeline(df, n_clusters=int(n_clusters), random_state=int(seed))
            out_df = dataframe_from_pipeline(
                result, df, cost_per_hour_vnd=float(cost_per_gio)
            )
            st.success(
                f"Đã gom **{len(out_df)}** hồ sơ thành **{result['n_clusters']}** nhóm. "
                f"(Embedding: {result['embedding_mode']})"
            )

            st.markdown(
                cluster_story_markdown(
                    df, result, cost_per_hour_vnd=float(cost_per_gio)
                )
            )

            st.subheader("Các cụm = biến thể (tên gọi điển hình)")
            bang_cum = cluster_variant_summary_table(out_df).rename(
                columns={
                    "cum_kmeans": "Cụm (K-Means)",
                    "ma_bien_the_dien_hinh": "Mã biến thể (điển hình)",
                    "chi_tiet_ma_bien_the": "Ý nghĩa / giải thích mã",
                    "ten_bien_the_dien_hinh": "Tên biến thể điển hình",
                    "so_ho_so": SO_HO_SO_LABEL,
                    "ghi_chu": "Ghi chú (trong cụm & bảng chi tiết)",
                }
            )
            st.caption(
                "**«Bảng chi tiết»** trong ghi chú = mục **Bảng chi tiết: từng hồ sơ — cụm — biến thể…** ngay bên dưới "
                "(mỗi dòng một hồ sơ: `case_id`, cụm, mã biến thể, chuỗi thao tác…)."
            )
            st.dataframe(bang_cum, use_container_width=True, hide_index=True)

            dem_bt = (
                out_df.groupby(["ma_bien_the", "ten_bien_the"], as_index=False)
                .size()
                .rename(columns={"size": "so_case_trong_log"})
                .sort_values("so_case_trong_log", ascending=False)
            )
            st.subheader("Danh mục biến thể trong toàn bộ log (trước khi gom cụm)")
            st.dataframe(
                dem_bt.rename(
                    columns={
                        "ma_bien_the": "Mã",
                        "ten_bien_the": "Tên biến thể",
                        "so_case_trong_log": "Số case",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            cols_detail = [
                "case_id",
                "cluster",
                "ma_bien_the",
                "ten_bien_the",
                "cycle_time_gio",
                "chi_phi_vnd",
                "chuoi_thao_tac",
                "goi_y",
            ]
            cols_ok = [c for c in cols_detail if c in out_df.columns]
            hien_thi = out_df.sort_values(["cluster", "case_id"])[cols_ok].rename(
                columns={
                    "case_id": "Mã hồ sơ",
                    "cluster": "Cụm",
                    "ma_bien_the": "Mã biến thể",
                    "ten_bien_the": "Tên biến thể",
                    "cycle_time_gio": "Chu kỳ (giờ)",
                    "chi_phi_vnd": "Chi phí ước tính (VNĐ)",
                    "chuoi_thao_tac": "Chuỗi thao tác (đúng thứ tự thời gian)",
                    "goi_y": "Gợi ý đọc nhanh",
                }
            )
            st.subheader("Bảng chi tiết: từng hồ sơ — cụm — biến thể — chu kỳ — chi phí — chuỗi thao tác")
            st.dataframe(hien_thi, use_container_width=True, hide_index=True)

            cc = cluster_cycle_cost_summary(out_df)
            if not cc.empty:
                st.subheader("Theo cụm: chu kỳ & chi phí (ước lượng)")
                st.dataframe(
                    cc.rename(
                        columns={
                            "cluster": "Cụm",
                            "so_ho_so": SO_HO_SO_LABEL,
                            "cycle_tb_gio": "TB chu kỳ (giờ)",
                            "cycle_tong_gio": "Tổng chu kỳ (giờ)",
                            "chi_phi_tong_vnd": "Tổng chi phí (VNĐ)",
                            "chi_phi_tb_vnd": "TB chi phí / hồ sơ (VNĐ)",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            vc_by_count_desc = out_df["cluster"].value_counts().sort_values(ascending=False)

            st.subheader("Cụm đông case nhất trước (thường là luồng phổ biến)")
            st.caption("Trục ngang: **từ trái sang phải** = từ cụm **đông nhất** đến **ít dần**.")
            cluster_count_bar_chart(vc_by_count_desc)

            with st.expander("Chi tiết kỹ thuật (số đo đồ thị & Silhouette)", expanded=False):
                m1, m2, m3 = st.columns(3)
                if result["silhouette"] is not None:
                    m1.metric(
                        "Silhouette",
                        f"{result['silhouette']:.3f}",
                        help="Gần 1 = các cụm tách bạch hơn (tham khảo).",
                    )
                else:
                    m1.metric("Silhouette", "Không tính được")
                m2.metric(SO_HO_SO_LABEL, len(out_df))
                m3.metric("Số cụm K", result["n_clusters"])
                ky_thuat = out_df.sort_values(["cluster", "case_id"])[
                    ["case_id", "cluster", "so_buoc", "nodes_do_thi", "edges_do_thi"]
                ].rename(
                    columns={
                        "so_buoc": "Số bước (activity)",
                        "nodes_do_thi": "Số nút đồ thị",
                        "edges_do_thi": "Số cạnh đồ thị",
                    }
                )
                st.dataframe(ky_thuat, use_container_width=True, hide_index=True)

            st.info(
                "**Mã BT-xx** là tên biến thể do hệ thống gán theo **mẫu chuỗi thao tác** (luồng chuẩn, có Check, "
                "Reject…). **Cụm K-Means** gom các embedding gần nhau — đôi khi một cụm chứa nhiều mã BT nếu K nhỏ; "
                "tăng K hoặc xem bảng “Danh mục biến thể”. Kết luận đúng/sai nghiệp vụ vẫn cần **quy định** của đơn vị."
            )

            st.session_state["pl_result"] = result
            st.session_state["pl_out_df"] = out_df
            st.session_state["pl_df"] = df
            try:
                save_graph_embedding_artifacts(result)
                st.info(
                    "📁 **Đã lưu embedding** vào **`outputs/`**: `graph2vec_graph_embeddings.csv`, "
                    "`graph2vec_graph_embeddings.npz`, `graph2vec_run_meta.json`. "
                    "Xem lại trong expander **📂 Graph embedding đã lưu…** ở đầu tab, hoặc mở file trực tiếp."
                )
            except OSError as exc_save:
                st.warning(f"Huấn luyện xong nhưng không ghi được `outputs/`: {exc_save}")
        except Exception as exc:
            st.error(f"Lỗi: {exc}")

    if "pl_result" in st.session_state:
        st.divider()
        res_v = st.session_state["pl_result"]
        out_v = st.session_state.get("pl_out_df")
        case_ids_v = res_v["case_ids"]
        emb_v = res_v["embeddings"]

        st.subheader("Graph embedding (Graph2Vec / fallback)")
        st.caption(
            f"**Phương pháp:** `{res_v['embedding_mode']}`. "
            "Sau mỗi lần train, bảng này đồng bộ với file **`outputs/graph2vec_graph_embeddings.csv`** (và `.npz`). "
            "Trong notebook **GNN**, *graph embedding* là tensor `(1, d)` cho **một** đồ thị; "
            "ở đây **mỗi hàng** = một case = một đồ thị succession."
        )
        n_c, n_d = int(emb_v.shape[0]), int(emb_v.shape[1])
        m1, m2, m3 = st.columns(3)
        m1.metric("Số case (đồ thị)", n_c)
        m2.metric("Số chiều embedding", n_d)
        m3.metric("Cụm K-Means", res_v["n_clusters"])
        df_emb = graph_embeddings_from_pipeline_result(res_v)
        d_csv, d_npz = st.columns(2)
        with d_csv:
            st.download_button(
                "Tải CSV — graph_embeddings.csv",
                data=graph_embeddings_csv_bytes(df_emb),
                file_name="graph2vec_graph_embeddings.csv",
                mime="text/csv",
                key="dl_emb_csv",
            )
        with d_npz:
            st.download_button(
                "Tải NPZ — case_ids + embeddings",
                data=graph_embeddings_npz_bytes(case_ids_v, emb_v),
                file_name="graph2vec_graph_embeddings.npz",
                mime="application/octet-stream",
                key="dl_emb_npz",
            )
        st.dataframe(df_emb, use_container_width=True, hide_index=True, height=280)
        peek = st.selectbox(
            "Xem vector đầy đủ một hồ sơ (in ra dạng mảng)",
            options=case_ids_v,
            key="emb_peek_case",
        )
        idx = case_ids_v.index(str(peek))
        vec = np.asarray(emb_v[idx], dtype=float)
        st.code(
            np.array2string(vec, precision=6, separator=", ", max_line_width=100),
            language=None,
        )

        st.divider()
        st.subheader("Trực quan hoá (Plotly, Pyvis)")
        st.caption(
            "**Plotly:** PCA embedding, heatmap cosine. **Pyvis:** đồ thị succession tương tác (kéo nút, zoom)."
        )
        cosine_dampen = st.slider(
            "Thu nhỏ cosine hiển thị (heatmap + bảng truy vấn)",
            min_value=0.5,
            max_value=1.0,
            value=0.78,
            step=0.02,
            key="cosine_dampen",
            help="Nhân cosine với hệ số này; đường chéo / chính case truy vấn vẫn = 1. Giảm số hiển thị, thứ tự gần giữ nguyên.",
        )
        cl_v = clusters_for_case_order(case_ids_v, out_v)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                figure_embedding_pca_scatter(emb_v, case_ids_v, cl_v),
                use_container_width=True,
            )
        with c2:
            hm_fig, hm_warn = figure_cosine_heatmap(
                emb_v, case_ids_v, cl_v, cosine_dampen=float(cosine_dampen)
            )
            if hm_warn:
                st.warning(hm_warn)
            st.plotly_chart(hm_fig, use_container_width=True)

        pl_df_v = st.session_state.get("pl_df")
        if pl_df_v is not None:
            g_case = st.selectbox(
                "Chọn hồ sơ để xem đồ thị succession (activity → activity)",
                options=case_ids_v,
                key="viz_succession_case",
            )
            st.markdown("**Đồ thị succession — Pyvis** (kéo, zoom)")
            components.html(
                succession_pyvis_html(pl_df_v, str(g_case)),
                height=500,
                scrolling=True,
            )
        else:
            st.caption(
                "Chạy lại **Huấn luyện** để lưu log gốc — khi đó có thể vẽ đồ thị succession tương tác."
            )

        st.divider()
        st.subheader("Lọc & tìm kiếm trùng lặp (Graph2Vec + cosine)")
        st.caption(
            "So khớp **gần** theo vector embedding (cosine; đã áp **hệ số thu nhỏ** ở slider phía trên). "
            "Nên xem thêm expander **chuỗi activity y hệt** — Graph2Vec không đảm bảo cosine ≈ 1 cho hai case trùng trace."
        )
        res = st.session_state["pl_result"]
        out_df_cached = st.session_state.get("pl_out_df")
        case_ids = res["case_ids"]
        emb = res["embeddings"]

        q_case = st.selectbox(
            "Chọn hồ sơ truy vấn (so với toàn kho)",
            options=case_ids,
            key="dup_query_case",
        )
        thr = st.slider(
            "Ngưỡng cảnh báo trùng / gần trùng (cosine)",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.01,
            help="Các hồ sơ khác có cosine ≥ ngưỡng được coi là gần trùng cấu trúc (minh họa).",
            key="dup_threshold",
        )

        pl_dup = st.session_state.get("pl_df")
        if pl_dup is not None:
            trace_pairs = exact_trace_duplicate_pairs(pl_dup)
            with st.expander(
                "Trùng theo **chuỗi activity** trong event log (từng cặp case)",
                expanded=False,
            ):
                if "process_id" in pl_dup.columns:
                    st.caption(
                        "Case **cùng `process_id`** là cùng một quy trình (2 bản thực thi). "
                        "Case **khác `process_id`** nhưng chuỗi activity giống hệt = **hai quy trình trùng luồng** "
                        "(khớp bảng cặp P↔P ở expander 20 quy trình)."
                    )
                else:
                    st.caption(
                        "Cặp case có cùng thứ tự activity. So với Graph2Vec/cosine: embedding không luôn phản ánh trùng tuyệt đối."
                    )
                if trace_pairs.empty:
                    st.write("Không có cặp case nào trùng hoàn toàn chuỗi activity.")
                else:
                    st.dataframe(
                        trace_pairs.rename(
                            columns={
                                "case_a": "Hồ sơ A",
                                "case_b": "Hồ sơ B",
                                "so_buoc": "Số bước",
                                "trace_preview": "Đầu trace (xem trước)",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        try:
            ranked = rank_similarity_to_query(
                case_ids, emb, q_case, cosine_dampen=float(cosine_dampen)
            )
            if out_df_cached is not None and "cluster" in out_df_cached.columns:
                cmap = out_df_cached[["case_id", "cluster"]].drop_duplicates()
                ranked = ranked.merge(cmap, on="case_id", how="left")

            hien = ranked.rename(
                columns={
                    "case_id": "Mã hồ sơ",
                    "cosine_similarity": "Cosine",
                    "tuong_dong_pct": "Tương đồng (%)",
                    "la_chinh_no": "Là chính nó",
                    "cluster": "Cụm",
                }
            )
            st.dataframe(
                hien,
                use_container_width=True,
                hide_index=True,
            )

            others = ranked[(~ranked["la_chinh_no"]) & (ranked["cosine_similarity"] >= thr)]
            if others.empty:
                st.success(
                    f"Không có hồ sơ nào khác đạt cosine ≥ **{thr:.0%}** so với **{q_case}**."
                )
            else:
                st.warning(
                    f"Cảnh báo (minh họa): **{len(others)}** hồ sơ có cấu trúc embedding gần **{q_case}** "
                    f"(cosine ≥ {thr:.0%}): {', '.join(others['case_id'].astype(str).tolist())}"
                )

            with st.expander("Cặp hồ sơ có độ tương đồng cao (toàn bộ kho)", expanded=False):
                pair_thr = st.slider(
                    "Ngưỡng cosine cho cặp",
                    min_value=0.85,
                    max_value=1.0,
                    value=0.95,
                    step=0.01,
                    key="pair_threshold",
                )
                pairs = pairs_exceeding_similarity(
                    case_ids, emb, pair_thr, cosine_dampen=float(cosine_dampen)
                )
                if pairs.empty:
                    st.write(f"Không có cặp nào đạt cosine ≥ **{pair_thr:.2f}**.")
                else:
                    st.dataframe(
                        pairs.rename(
                            columns={
                                "case_a": "Hồ sơ A",
                                "case_b": "Hồ sơ B",
                                "cosine_similarity": "Cosine",
                                "tuong_dong_pct": "Tương đồng (%)",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        except Exception as exc:
            st.error(f"Lỗi khi tính similarity: {exc}")


def main() -> None:
    st.set_page_config(page_title="Process Variant Clustering Demo", page_icon="🔎", layout="wide")

    st.title("🔎 Bank20 — Phát hiện biến thể từ Event Log")
    st.caption(
        "Đề tài: Phát hiện và phân cụm biến thể quy trình ngân hàng từ event log — Graph2Vec + K-Means"
    )

    tab_diagrams, tab_pipeline = st.tabs(["📐 Sơ đồ BPMN & Petri net", "🔬 Event log & pipeline"])
    with tab_diagrams:
        render_diagrams_web_tab()
    with tab_pipeline:
        render_event_log_thesis_tab()


if __name__ == "__main__":
    main()
