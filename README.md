# Detecting and Clustering Process Variants from Event Logs (Graph2Vec + K-Means)

Ung dung Streamlit demo cho de tai: **Phat hien va phan cum bien the quy trinh tu nhat ky su kien**, ket hop Process Mining (do thi tu trace) + Graph2Vec + K-Means. **Du lieu demo** mo phong **Loan Origination (LO)** — giai ngan tin dung / xet duyet khoan vay.

## 1) Cai dat

Dùng lệnh **`python`** (Conda / pyenv thường có sẵn). Nếu máy chỉ có `python3`, thay tương ứng **chỉ khi tạo venv lần đầu**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2) Chay app

```bash
python -m streamlit run app.py
```

(Cách `python -m streamlit` luôn đúng với interpreter đang bật, không cần `streamlit` trên PATH.)

App chi gom 1 man hinh Event Log → Graph2Vec + K-Means de dung trong tam de tai.

## 3) Ma bien the (BT-xx) — ngu canh LO

Moi trace duoc gan **ma + ten bien the** theo mau **Loan Origination** (vi du BT-01 lo chuan 5 buoc, BT-02 co CollateralCheck, BT-03 bo CreditScoring, …). Cum K-Means hien **ten bien the dien hinh** trong tung cum.

## 4) Du lieu demo

File mau: `data/event_log_demo.csv`

- Cot bat buoc: `case_id`, `activity`, `timestamp`
- Khoang **36 case**, **~186 dong** su kien (activity bang tieng Anh ngan gon cho slide/bao cao):
  - **LO chuan** (da so): `LoanApplication → KYC → CreditScoring → CreditApproval → Disbursement`
  - **Bien the pho bien**: them `CollateralCheck` (TSDB) truoc phê duyet
  - **Bien the lech**: bo `CreditScoring` (nhay coc toi `CreditApproval`)
  - **Bien the hiem** (it case): rut gon `WithdrawnEarly`; lap `CreditScoring`; `LegalReview`; `CreditRejection` + `AccountClosed`; `CollateralCheck` + `LegalReview`; `EscalateCase` khong qua `CreditScoring`

Ban co the upload CSV rieng (tat checkbox "Dung file demo").

## 5) Pipeline (2 model)

1. **Trich do thi**: moi case = mot do thi vo huong, canh noi hai activity lien tiep theo thoi gian (direct succession).
2. **Graph2Vec** (`karateclub`): hoc embedding; neu khong cai duoc thi fallback embedding thong ke cau truc.
3. **K-Means**: phan cum cac vector embedding.

Giao dien hien: bang Case → Cluster, bieu do so case/cum, Silhouette (neu tinh duoc).

**Chu ky (cycle time) & chi phi uoc luong:** voi moi `case_id`, **chu ky** = `timestamp` su kien cuoi - su kien dau (end-to-end). **Chi phi** = `chu_ky_gio * don_gia_VND/gio` (nhap tren app) — mo hinh minh hoa; thay bang so lieu that cua don vi khi bao cao.

## 6) Ghi chu

- Ket qua phan cum phan anh **mien giong nhau ve cau truc trace**, khong tu dong la **cung nghiep vu**.
- Graph2Vec can `karateclub`; neu loi cai dat, he thong van chay nho embedding fallback.

## 7) Mau BPMN vs Petri net (LO chuan)

**Tren web (Streamlit):** chay app → tab **« Sơ đồ BPMN & Petri net »** — xem hai hinh truc tiep trong trinh duyet.

**Hoac mo file HTML:** `docs/diagrams/view.html` (mo bang trinh duyet, hoac `cd docs/diagrams && python -m http.server` roi vao `http://localhost:8000/view.html`).

Hai file SVG (chen Word/LaTeX hoac mo rieng):

- `docs/diagrams/bpmn_lo_standard.svg` — BPMN 2.0: Start → 5 task → End (happy path giong trace chuan trong CSV).
- `docs/diagrams/petrinet_lo_standard.svg` — Petri net dang **workflow net**: place–transition–place … (token o p0).
