# 🍷 Phân loại loại rượu dựa trên 13 đặc trưng hóa học

## 👋 Giới thiệu và mục tiêu
- Tập dữ liệu này chứa thông tin về 13 đặc trưng hóa học của rượu vang và loại rượu tương ứng (class).
- Mục tiêu là xây dựng mô hình phân loại để dự đoán loại rượu dựa trên các đặc trưng này.
- Cấu trúc của bài gồm có các bước sau:
    1. Khám phá và tiền xử lý dữ liệu
    2. Xây dựng mô hình Machine Learning
    3. Huấn luyện và phân tích lỗi
    4. Tối ưu hóa mô hình
    5. Triển khai mô hình
    6. Đánh giá tổng quan và đề xuất cải tiến

## 📂 Giới thiệu bộ dữ liệu

Bộ dữ liệu được sử dụng là [Wine dataset trên Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/wine-dataset), bao gồm các thông tin:

- 13 đặc trưng hóa học của rượu, bao gồm:
  - Alcohol
  - Malic acid
  - Ash
  - Alcalinity of ash
  - Magnesium
  - Total phenols
  - Flavanoids
  - Nonflavanoid phenols
  - Proanthocyanins
  - Color intensity
  - Hue
  - OD280/OD315 of diluted wines
  - Proline
- Nhãn phân loại loại rượu (Wine Class): 1, 2 hoặc 3

---

## 🧭 Chi tiết về các bước thực hiện

1. **Khám phá dữ liệu (EDA) và tiền xử lý dữ liệu**
   - Khám phá dữ liệu (EDA)
      - Đọc và hiển thị các dòng đầu tiên của dữ liệu
      - Hiển thị thông tin và cấu trúc dữ liệu
      - Kiểm tra giá trị null
      - Phân tích thống kê mô tả (mean, std, min, max, etc.) và đếm số lượng mẫu theo từng lớp
      - Trực quan hóa dữ liệu
   - Tiền xử lý dữ liệu 
      - Chia dữ liệu thành tập huấn luyện và tập kiểm tra (train-test split)  
      - Chia dữ liệu thành các đặc trưng (X) và nhãn (y)  
      - Chia dữ liệu thành các tập con cho huấn luyện, kiểm tra và xác thực (train, test, validation)  
      - Chuyển đổi kiểu dữ liệu nếu cần thiết  
      - Kiểm tra và xử lý các giá trị ngoại lệ (outliers)

2. **Xây dựng mô hình học máy**
   - Logistic Regression
   - Navie Bayes
   - Đánh giá mô hình và so sánh hiệu suất giữa các mô hình
   
3. **Huấn luyện và phân tích lỗi**  
   - Đánh giá accuracy, precision, recall, F1-score
   - Confusion Matrix
   - Cross-validation

4. **Tối ưu mô hình**  
   - Sử dụng GridSearchCV để tìm siêu tham số tối ưu
   - So sánh Logistic Regression và Navie Bayes ở các chỉ số F1 - score

5. **Triển khai mô hình**  
   - Lưu mô hình sau tối ưu
   - Dự đoán rượu khi người dùng nhập vào các thông số hóa học

6. **Đánh giá tổng quan và đề xuất cải tiến**
   - Đánh giá tổng quan về mô hình
   - Đề xuất cải tiến cho mô hình
   - Đề xuất các hướng nghiên cứu tiếp theo

---

## 📊 Kết quả mong đợi

- Mô hình phân loại rượu có độ chính xác cao
- Hiểu được các yếu tố hóa học ảnh hưởng đến phân loại rượu
- So sánh hiệu năng giữa các mô hình khác nhau

---

## 📚 Tham khảo

- [Wine dataset trên Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/wine-dataset)
- Siphendulwe Zaza et al., "Wine feature importance and quality prediction: A comparative study of machine learning algorithms with unbalanced data", 2023.  
  [arxiv.org/abs/2310.01584](https://arxiv.org/abs/2310.01584)
- S. Di and Y. Yang, "Prediction of Red Wine Quality Using One-dimensional Convolutional Neural Networks", 2022.  
  [arxiv.org/abs/2208.14008](https://arxiv.org/abs/2208.14008)
