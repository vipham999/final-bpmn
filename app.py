import io
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SO_HO_SO_LABEL = "Số hồ sơ"
_ROOT = Path(__file__).resolve().parent
_DIAGRAM_BPMN = _ROOT / "docs" / "diagrams" / "bpmn_lo_standard.svg"
_DIAGRAM_PN = _ROOT / "docs" / "diagrams" / "petrinet_lo_standard.svg"


def render_diagrams_web_tab() -> None:
    """Hiển thị BPMN & Petri net (SVG) trên web — cùng nội dung file trong docs/diagrams/."""
    st.subheader("BPMN 2.0 — Loan Origination (happy path)")
    st.caption(
        "Start → LoanApplication → KYC → CreditScoring → CreditApproval → Disbursement → End. "
        "Khớp trace chuẩn trong `data/event_log_demo.csv`."
    )
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
        "Hai file gốc: `docs/diagrams/bpmn_lo_standard.svg`, `docs/diagrams/petrinet_lo_standard.svg` — "
        "có thể mở bằng trình duyệt hoặc chèn vào báo cáo."
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

from event_log_pipeline import (
    cluster_cycle_cost_summary,
    cluster_story_markdown,
    cluster_variant_summary_table,
    dataframe_from_pipeline,
    demo_csv_path,
    load_event_log_csv,
    run_pipeline,
)

def render_event_log_thesis_tab() -> None:
    st.subheader("Loan Origination (LO) — Event log → đồ thị → Graph2Vec → K-Means")
    st.markdown(
        """
**Kịch bản demo:** quy trình **giải ngân tín dụng / xét duyệt khoản vay** (Loan Origination). Mỗi `case_id` là một hồ sơ vay; `activity` là các bước xử lý (KYC, scoring, phê duyệt, giải ngân…).

1. **Trích đồ thị**: với mỗi `case_id`, sắp xếp theo `timestamp`, nối cạnh giữa activity liền kề (succession trực tiếp).
2. **Model 1 – Graph2Vec**: `fit` trên đồ thị của từng hồ sơ → vector embedding.
3. **Model 2 – K-Means**: phân cụm các vector → gom **biến thể LO** có cấu trúc tương đồng.
"""
    )
    with st.expander("Bạn sẽ đọc kết quả thế nào? (không chỉ là con số)", expanded=False):
        st.markdown(
            """
- **Cột “Cụm” (cluster)**: máy gom các **hồ sơ (case)** có **cách đi quy trình giống nhau** vào một nhóm. Cùng cụm ≠ cùng đúng/sai nghiệp vụ, mà là **cùng kiểu luồng**.
- **Chuỗi thao tác**: đúng thứ tự **activity** trong hồ sơ vay — dễ hiểu nhất; “gợi ý” là mô tả nhanh theo bước LO (KYC, CreditScoring, CollateralCheck…).
- **Silhouette** (nếu có): đo độ “tách bạch” giữa các cụm (càng gần **1** thường càng rõ ràng). Chỉ mang tính **tham khảo**, không phải kết luận cuối.
- **Node/edge đồ thị**: dùng nội bộ để so khớp cấu trúc; có thể xem trong phần kỹ thuật.
"""
        )

    use_demo = st.checkbox("Dùng file demo mẫu (data/event_log_demo.csv)", value=True)
    st.caption(
        "File demo **Loan Origination**: **36 hồ sơ**. Đa số là **LO chuẩn** "
        "(LoanApplication → KYC → CreditScoring → CreditApproval → Disbursement); "
        "**C026–C036** là biến thể hiếm (rút gọn, lặp scoring, pháp chế, từ chối, leo cấp…) để thấy **cụm nhỏ**."
    )
    uploaded = None
    if not use_demo:
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
            if use_demo:
                df = load_event_log_csv(demo_csv_path())
            else:
                if uploaded is None:
                    st.warning("Hãy upload file CSV hoặc bật dùng file demo.")
                    return
                df = load_event_log_csv(io.BytesIO(uploaded.read()))

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
                    "ma_bien_the_dien_hinh": "Mã biến thể",
                    "ten_bien_the_dien_hinh": "Tên biến thể điển hình",
                    "so_ho_so": SO_HO_SO_LABEL,
                    "ghi_chu": "Ghi chú",
                }
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
        except Exception as exc:
            st.error(f"Lỗi: {exc}")


def main() -> None:
    st.set_page_config(page_title="Process Variant Clustering Demo", page_icon="🔎", layout="wide")

    st.title("🔎 Loan Origination — Phát hiện biến thể từ Event Log")
    st.caption(
        "Đề tài: Phát hiện và phân cụm biến thể quy trình (LO) từ event log — Graph2Vec + K-Means"
    )

    tab_diagrams, tab_pipeline = st.tabs(["📐 Sơ đồ BPMN & Petri net", "🔬 Event log & pipeline"])
    with tab_diagrams:
        render_diagrams_web_tab()
    with tab_pipeline:
        render_event_log_thesis_tab()


if __name__ == "__main__":
    main()
