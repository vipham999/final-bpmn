Lựa chọn tên đề tài "Xây dựng công cụ lọc và tìm kiếm quy trình trùng lặp bằng thuật toán Graph2Vec" rất xuất sắc, vừa mang tính học thuật vừa thể hiện rõ giá trị thực tiễn là ứng dụng truy vấn (Process Retrieval) và phân loại biến thể (Process Variant Clustering)
.
Vì bạn chỉ có 2 tuần để làm bản Demo (Proof of Concept), dưới đây là lộ trình 14 ngày "thần tốc" được thiết kế tối giản, tập trung thẳng vào cốt lõi của thuật toán Graph2Vec:
Tuần 1: Khởi tạo dữ liệu và Huấn luyện mô hình (Ngày 1 - Ngày 7)
Ngày 1 - 2: Cài đặt môi trường và tìm hiểu thuật toán
Công việc: Cài đặt Python và các thư viện cần thiết gồm networkx (để tạo đồ thị), karateclub (để chạy Graph2Vec) và scikit-learn (để tính khoảng cách vector)
.
Lý thuyết cần nắm: Nắm vững nguyên lý Graph2Vec biến toàn bộ một đồ thị quy trình thành một vector duy nhất dựa trên cấu trúc hình học (topo) của nó, thay vì quan tâm đến thứ tự tuyến tính hay tên gọi từng bước
.
Ngày 3 - 4: Dựng dữ liệu giả lập (Hardcode Graphs)
Công việc: Thay vì viết code đọc file XML phức tạp, bạn dùng thư viện networkx tạo thủ công 4-5 đồ thị đại diện cho các quy trình trong kho dữ liệu
.
Chiến thuật tạo data:
Quy trình 1 (Quy trình chuẩn gốc): Tạo một đồ thị tuần tự (ví dụ dùng nx.cycle_graph)
.
Quy trình 2 (Quy trình trùng lặp/Biến thể nhẹ): Tạo một đồ thị gần giống Quy trình 1, chỉ thêm hoặc bớt 1 nhánh nhỏ
.
Quy trình 3 (Quy trình hoàn toàn khác): Tạo đồ thị có cấu trúc hình sao (nx.star_graph) để làm mốc đối chứng
.
Ngày 5 - 7: Nhúng đồ thị (Graph Embedding) bằng Graph2Vec
Công việc: Đưa danh sách các đồ thị vừa tạo vào mô hình Graph2Vec.
Thực thi:
Khởi tạo mô hình: model = Graph2Vec(dimensions=16)
.
Huấn luyện cấu trúc: Gọi hàm model.fit(graphs)
.
Trích xuất vector: Dùng embeddings = model.get_embedding() để lấy các vector số học đại diện cho từng quy trình
.

---

Tuần 2: Xây dựng bộ lọc trùng lặp và Chuẩn bị Demo (Ngày 8 - Ngày 14)
Ngày 8 - 9: Viết thuật toán lọc và tìm kiếm trùng lặp
Công việc: Sử dụng hàm cosine_similarity từ thư viện sklearn.metrics.pairwise để tính toán độ tương đồng giữa các vector
.
Áp dụng logic nghiệp vụ: Viết một hàm so sánh vector của "Quy trình truy vấn" với toàn bộ kho vector. Đặt một ngưỡng (Threshold) trùng lặp. Ví dụ: Nếu độ tương đồng Cosine > 90%, hệ thống sẽ in ra kết quả: "Cảnh báo: Quy trình này đã trùng lặp với Quy trình X trong hệ thống"
.
Ngày 10 - 11: Đóng gói thành công cụ Demo (Mini App)
Công việc: Xây dựng một giao diện dòng lệnh (CLI) đơn giản hoặc dùng thư viện Streamlit để tạo nhanh một giao diện web trong 1 buổi.
Kịch bản Demo: Người dùng chọn 1 quy trình mẫu → Nhấn nút "Quét trùng lặp" → Màn hình hiện ra % tương đồng cấu trúc với các quy trình khác
.
Ngày 12 - 14: Viết báo cáo và Chuẩn bị Slide bảo vệ
Công việc: Tổng hợp lại quá trình làm thành slide.
Điểm nhấn khi thuyết trình: Nhấn mạnh rằng công cụ này giải quyết được bài toán đối chiếu cấu trúc quy trình mà không bị đánh lừa bởi tên gọi các bước (ví dụ: chi nhánh A gọi là "Tiếp nhận", chi nhánh B gọi là "Khởi tạo" nhưng cấu trúc giống nhau thì máy vẫn phát hiện được)
. Khẳng định giá trị của công cụ là giúp ngân hàng/doanh nghiệp hợp nhất các quy trình rời rạc, từ đó giảm chi phí vận hành
.
Với lộ trình này, bạn hoàn toàn có thể hoàn thành một bản Demo ấn tượng và trực quan chỉ với khoảng 50 dòng code Python cốt lõi!
