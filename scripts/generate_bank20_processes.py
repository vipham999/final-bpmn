"""
Sinh:
- data/bank20_process_catalog.csv — 20 quy trình nghiệp vụ (mỗi dòng = một quy trình + chuỗi activity).
- data/event_log_bank20.csv — event log: nhiều case / quy trình, cột process_id để đối chiếu.

Một số quy trình cố ý trùng luồng (cùng activity_sequence) để minh họa phát hiện trùng.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT_OUT = ROOT / "data" / "bank20_process_catalog.csv"
LOG_OUT = ROOT / "data" / "event_log_bank20.csv"

# 20 quy trình: process_id, mã, tên VN, nhóm, chuỗi activity (|)
PROCESSES: list[dict[str, str]] = [
    {
        "process_id": "P01",
        "process_code": "ONB_FULL",
        "name_vn": "Mở khách hàng đầy đủ (KYC → rủi ro → tài khoản → thẻ)",
        "group_vn": "Onboarding",
        "activity_sequence": "CustomerIdentification|RiskProfileAssessment|AccountOpening|CardIssuance",
    },
    {
        "process_id": "P02",
        "process_code": "ONB_FAST",
        "name_vn": "Mở tài khoản nhanh + thông báo",
        "group_vn": "Onboarding",
        "activity_sequence": "CustomerIdentification|AccountOpening|CustomerNotification",
    },
    {
        "process_id": "P03",
        "process_code": "LEND_COLL",
        "name_vn": "Cho vay có tài sản đảm bảo",
        "group_vn": "Tín dụng",
        "activity_sequence": "LoanApplicationProcessing|CustomerIdentification|CreditScoring|CreditApproval|CollateralValuation|Disbursement",
    },
    {
        "process_id": "P04",
        "process_code": "LEND_NOCOLL",
        "name_vn": "Cho vay không tài sản đảm bảo",
        "group_vn": "Tín dụng",
        "activity_sequence": "LoanApplicationProcessing|CustomerIdentification|CreditScoring|CreditApproval|Disbursement",
    },
    {
        "process_id": "P05",
        "process_code": "LC_DISB",
        "name_vn": "Thư tín dụng → định giá → giải ngân",
        "group_vn": "Thương mại",
        "activity_sequence": "LetterOfCreditIssuance|CustomerIdentification|CollateralValuation|Disbursement",
    },
    {
        "process_id": "P06",
        "process_code": "FX_COMPL",
        "name_vn": "Ngoại tệ + giám sát + AML + đối soát",
        "group_vn": "Giao dịch & tuân thủ",
        "activity_sequence": "ForeignExchange|TransactionMonitoring|AMLScreening|Reconciliation",
    },
    {
        "process_id": "P07",
        "process_code": "CASH_XFER",
        "name_vn": "Tiền mặt → giám sát → chuyển khoản → đối soát",
        "group_vn": "Giao dịch",
        "activity_sequence": "CashDepositWithdrawal|TransactionMonitoring|FundTransfer|Reconciliation",
    },
    {
        "process_id": "P08",
        "process_code": "DEBT",
        "name_vn": "Thu nợ + KYC + AML + kiểm toán",
        "group_vn": "Tín dụng",
        "activity_sequence": "DebtCollection|CustomerIdentification|AMLScreening|InternalAudit",
    },
    {
        "process_id": "P09",
        "process_code": "AUDIT_TRAIL",
        "name_vn": "Giám sát giao dịch → đối soát → kiểm toán nội bộ",
        "group_vn": "Tuân thủ",
        "activity_sequence": "TransactionMonitoring|Reconciliation|InternalAudit",
    },
    {
        "process_id": "P10",
        "process_code": "PAY_SETTLE",
        "name_vn": "Chuyển tiền → quyết toán → thông báo KH",
        "group_vn": "Giao dịch",
        "activity_sequence": "FundTransfer|PaymentSettlement|CustomerNotification",
    },
    {
        "process_id": "P11",
        "process_code": "LEND_NOTIFY",
        "name_vn": "Cho vay (không TSĐB) + thông báo sau giải ngân",
        "group_vn": "Tín dụng",
        "activity_sequence": "LoanApplicationProcessing|CustomerIdentification|CreditScoring|CreditApproval|Disbursement|CustomerNotification",
    },
    # Trùng luồng với P03 (cùng nghiệp vụ “vay có TSĐB” — hai mã quy trình song song)
    {
        "process_id": "P12",
        "process_code": "LEND_COLL_DUP",
        "name_vn": "Cho vay có TSĐB (bản trùng quy trình P03 — ví dụ tài liệu song song)",
        "group_vn": "Tín dụng",
        "activity_sequence": "LoanApplicationProcessing|CustomerIdentification|CreditScoring|CreditApproval|CollateralValuation|Disbursement",
    },
    # Trùng luồng với P06
    {
        "process_id": "P13",
        "process_code": "FX_COMPL_ALIAS",
        "name_vn": "Tuân thủ FX (cùng luồng với P06)",
        "group_vn": "Giao dịch & tuân thủ",
        "activity_sequence": "ForeignExchange|TransactionMonitoring|AMLScreening|Reconciliation",
    },
    # Trùng luồng với P01
    {
        "process_id": "P14",
        "process_code": "ONB_FULL_COPY",
        "name_vn": "Onboarding đầy đủ (trùng P01)",
        "group_vn": "Onboarding",
        "activity_sequence": "CustomerIdentification|RiskProfileAssessment|AccountOpening|CardIssuance",
    },
    {
        "process_id": "P15",
        "process_code": "LC_SHORT",
        "name_vn": "LC rút gọn → giải ngân",
        "group_vn": "Thương mại",
        "activity_sequence": "LetterOfCreditIssuance|Disbursement",
    },
    {
        "process_id": "P16",
        "process_code": "CASH_DEEP",
        "name_vn": "Giao dịch tiền mặt + AML sâu + đối soát + kiểm toán",
        "group_vn": "Tuân thủ",
        "activity_sequence": "CashDepositWithdrawal|AMLScreening|TransactionMonitoring|Reconciliation|InternalAudit",
    },
    {
        "process_id": "P17",
        "process_code": "LEND_RESCORE",
        "name_vn": "Cho vay với chấm điểm hai lần",
        "group_vn": "Tín dụng",
        "activity_sequence": "LoanApplicationProcessing|CustomerIdentification|CreditScoring|CreditScoring|CreditApproval|Disbursement",
    },
    {
        "process_id": "P18",
        "process_code": "XFER_FX",
        "name_vn": "Chuyển tiền → FX → đối soát",
        "group_vn": "Giao dịch",
        "activity_sequence": "FundTransfer|ForeignExchange|Reconciliation",
    },
    {
        "process_id": "P19",
        "process_code": "LC_MON",
        "name_vn": "LC + giám sát + AML + giải ngân",
        "group_vn": "Thương mại",
        "activity_sequence": "LetterOfCreditIssuance|TransactionMonitoring|AMLScreening|Disbursement",
    },
    {
        "process_id": "P20",
        "process_code": "ACC_PAY",
        "name_vn": "Mở TK → chuyển tiền → quyết toán",
        "group_vn": "Onboarding & giao dịch",
        "activity_sequence": "AccountOpening|FundTransfer|PaymentSettlement",
    },
]

REPLICAS_PER_PROCESS = 5  # số case / quy trình (tăng để cụm tương đồng rõ hơn)


def write_catalog() -> None:
    import csv

    CAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "process_id",
        "process_code",
        "name_vn",
        "group_vn",
        "activity_sequence",
    ]
    with CAT_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in PROCESSES:
            w.writerow({k: p[k] for k in fields})


def write_event_log() -> None:
    lines = ["case_id,activity,timestamp,process_id"]
    t0 = datetime(2025, 2, 1, 8, 0, 0)
    step = 0
    for p in PROCESSES:
        seq = p["activity_sequence"].split("|")
        pid = p["process_id"]
        for r in range(1, REPLICAS_PER_PROCESS + 1):
            case_id = f"{pid}-C{r:02d}"
            ts = t0 + timedelta(hours=step)
            for act in seq:
                lines.append(f"{case_id},{act},{ts.isoformat(timespec='seconds')},{pid}")
                ts += timedelta(minutes=6)
            step += 1

    LOG_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_catalog()
    write_event_log()
    n_cases = len(PROCESSES) * REPLICAS_PER_PROCESS
    n_events = sum(len(p["activity_sequence"].split("|")) for p in PROCESSES) * REPLICAS_PER_PROCESS
    print(f"Wrote {CAT_OUT.name} ({len(PROCESSES)} processes)")
    print(f"Wrote {LOG_OUT.name} ({n_cases} cases, {n_events} events)")


if __name__ == "__main__":
    main()
