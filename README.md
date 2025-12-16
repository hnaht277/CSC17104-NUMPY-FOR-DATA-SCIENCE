# Airbnb NYC Price Prediction with NumPy

> Dự đoán giá thuê Airbnb tại New York City sử dụng Linear Regression được implement hoàn toàn bằng NumPy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Method](#-method)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)
- [Contact](#-contact)
- [License](#-license)

---

## Giới thiệu

### Mô tả bài toán

Dự án này tập trung vào việc **dự đoán giá thuê của các listing Airbnb tại New York City** dựa trên các đặc điểm như vị trí địa lý, loại phòng, số lượng đánh giá, và tính khả dụng. Đây là bài toán regression điển hình trong lĩnh vực Data Science và Machine Learning.

### Động lực và ứng dụng thực tế

**Tại sao bài toán này quan trọng?**

- **Cho chủ nhà (Hosts):** Xác định mức giá tối ưu để cạnh tranh và tối đa hóa doanh thu
- **Cho khách thuê:** Đánh giá xem giá listing có hợp lý so với thị trường
- **Cho nhà đầu tư:** Phân tích tiềm năng ROI của các khu vực khác nhau
- **Cho Airbnb platform:** Đề xuất giá tự động và phát hiện pricing anomalies

**Ứng dụng thực tế:**
- Dynamic pricing tools
- Market analysis dashboards  
- Investment opportunity identification
- Automated pricing recommendations

### Mục tiêu cụ thể

1. **Implement Linear Regression từ đầu** sử dụng thuần NumPy (không dùng scikit-learn)
2. **Áp dụng đầy đủ quy trình Data Science:** EDA → Preprocessing → Modeling → Evaluation
3. **Phân tích và trả lời các câu hỏi nghiệp vụ:**
   - Host nào bận rộn nhất và tại sao?
   - Sự khác biệt về lưu lượng giữa các khu vực?
   - Yếu tố nào ảnh hưởng mạnh nhất đến giá?
4. **Đạt performance chấp nhận được:** RMSE < $50, R² > 0.5
5. **Cung cấp insights và recommendations** cho các stakeholders

---

## Dataset

### Nguồn dữ liệu

- **Tên Dataset:** Airbnb NYC 2019
- **Nguồn:** [Kaggle - New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

### Mô tả các features

Dataset gồm **16 features** với các thông tin chi tiết:

| Feature | Kiểu dữ liệu | Mô tả | Ví dụ |
|---------|--------------|-------|-------|
| `id` | Integer | ID duy nhất của listing | 2539 |
| `name` | String | Tên listing | "Clean & quiet apt home by the park" |
| `host_id` | Integer | ID của host | 2787 |
| `host_name` | String | Tên host | "John" |
| `neighbourhood_group` | Categorical | Quận (Manhattan, Brooklyn, Queens...) | "Brooklyn" |
| `neighbourhood` | Categorical | Khu vực cụ thể | "Kensington" |
| `latitude` | Float | Vĩ độ | 40.64749 |
| `longitude` | Float | Kinh độ | -73.97237 |
| `room_type` | Categorical | Loại phòng (Entire home/apt, Private, Shared) | "Private room" |
| **`price`** | **Integer** | **Giá thuê/đêm (USD) - Target variable** | **149** |
| `minimum_nights` | Integer | Số đêm tối thiểu | 1 |
| `number_of_reviews` | Integer | Tổng số reviews | 9 |
| `last_review` | Date | Ngày review cuối cùng | 2019-05-21 |
| `reviews_per_month` | Float | Số reviews trung bình/tháng | 0.21 |
| `calculated_host_listings_count` | Integer | Số listing của host | 6 |
| `availability_365` | Integer | Số ngày available trong năm | 365 |

### Kích thước và đặc điểm dữ liệu

**Kích thước:**
- **Số samples:** 48,895 listings
- **Số features:** 16 cột
- **Dung lượng:** ~5 MB

**Đặc điểm quan trọng:**

1. **Missing Values:**
   - `name`: ~16 missing
   - `host_name`: ~21 missing
   - `last_review`: ~10,052 missing (20.5%)
   - `reviews_per_month`: ~10,052 missing (20.5%)

2. **Phân phối Price:**
   - Mean: $152.72
   - Median: $106
   - Std: $240.15
   - Range: $0 - $10,000
   - **Highly skewed** với nhiều outliers

3. **Phân phối Categorical:**
   - **Room Type:** Entire home/apt (52%), Private room (45.7%), Shared room (2.4%)
   - **Neighbourhood Group:** Manhattan (44.3%), Brooklyn (41.1%), Queens (9.7%), Bronx (2.3%), Staten Island (0.8%)

4. **Challenges:**
   - Outliers cực lớn trong price ($10,000)
   - Missing values đáng kể trong reviews
   - Imbalanced distribution giữa các neighbourhood groups

---

## Method

### Quy trình xử lý dữ liệu

```
Raw Data → EDA → Preprocessing → Feature Engineering → Modeling → Evaluation
```

#### **1. Exploratory Data Analysis (EDA)**

- Load data bằng NumPy và CSV module
- Thống kê mô tả: mean, median, std, quartiles
- Phân tích missing values
- Visualization: histograms, box plots, correlation matrix
- Phân tích theo categorical features (room_type, neighbourhood_group)

#### **2. Data Preprocessing**

**Xử lý Missing Values:**
```python
# Reviews_per_month: Điền 0 (listing chưa có reviews)
# Các cột số khác: Điền median (robust với outliers)
# Price, location: Loại bỏ rows (thông tin bắt buộc)
```

**Xử lý Outliers:**
```python
# Loại bỏ: price > $1000 (luxury segment, không đại diện)
# Loại bỏ: minimum_nights > 365 (có thể là lỗi data)
```

**Feature Engineering:**
- One-hot encoding: `room_type` → 3 binary features
- One-hot encoding: `neighbourhood_group` → 5 binary features
- Feature mới: `review_frequency` = `number_of_reviews` / (`reviews_per_month` + 1)

**Normalization:**
- Min-Max Scaling: $(x - x_{min}) / (x_{max} - x_{min})$
- Áp dụng cho tất cả features số

#### **3. Train/Test Split**

- Train: 80% (random shuffle với seed=42)
- Test: 20%
- Đảm bảo reproducibility

### Thuật toán sử dụng

#### **Linear Regression with Gradient Descent**

**Giả thuyết (Hypothesis):**

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T x$$

Trong đó:
- $\theta$ = vector hệ số (weights + bias)
- $x$ = vector features
- $n$ = số lượng features

**Loss Function (Mean Squared Error):**

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Với regularization L2 (Ridge):

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

**Gradient Descent Update Rule:**

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

Cụ thể:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m}\theta_j$$

Trong đó:
- $\alpha$ = learning rate (0.1)
- $\lambda$ = regularization parameter (0.01)
- $m$ = số samples

### Cách implement bằng NumPy

#### **Matrix Operations**

```python
# Forward pass (Prediction)
y_pred = X @ weights + bias  # Matrix multiplication: (m, n) @ (n, 1) = (m, 1)

# Gradient computation
dw = (1/m) * (X.T @ (y_pred - y_true))  # (n, m) @ (m, 1) = (n, 1)
db = (1/m) * np.sum(y_pred - y_true)    # Scalar

# Add L2 regularization to gradient
dw += (lambda_/m) * weights

# Update parameters
weights -= learning_rate * dw
bias -= learning_rate * db
```

#### **Key NumPy Techniques**

1. **Broadcasting:** Tự động mở rộng dimensions cho phép operations giữa arrays khác size
2. **Vectorization:** Thay thế loops bằng array operations (nhanh hơn 10-100x)
3. **Boolean Indexing:** Filter data hiệu quả: `data[data[:, price_col] < 1000]`
4. **Array Stacking:** Kết hợp features: `np.hstack([feature1, feature2])`

#### **Evaluation Metrics Implementation**

```python
# Mean Squared Error
MSE = np.mean((y_true - y_pred) ** 2)

# Root Mean Squared Error
RMSE = np.sqrt(MSE)

# Mean Absolute Error
MAE = np.mean(np.abs(y_true - y_pred))

# R² Score
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
R2 = 1 - (ss_res / ss_tot)
```

#### **K-Fold Cross Validation**

```python
# Chia dữ liệu thành k folds
n_samples = len(X)
indices = np.arange(n_samples)
np.random.shuffle(indices)

fold_size = n_samples // k_folds
for i in range(k_folds):
    val_start = i * fold_size
    val_end = (i + 1) * fold_size if i < k_folds - 1 else n_samples
    
    val_indices = indices[val_start:val_end]
    train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
    
    # Train và evaluate cho mỗi fold
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 hoặc cao hơn
- pip package manager
- Jupyter Notebook hoặc JupyterLab

### Thư viện yêu cầu

```
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### Các bước cài đặt

#### 1. Clone repository

```bash
git clone https://github.com/hnaht277/CSC17104-NUMPY-FOR-DATA-SCIENCE.git
cd CSC17104-NUMPY-FOR-DATA-SCIENCE
```

#### 2. Tạo virtual environment (khuyến nghị)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

#### 4. Kiểm tra cài đặt

```bash
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

---

## Usage

### Hướng dẫn chạy từng phần

Project được chia thành 3 notebooks chính, nên chạy theo thứ tự:

#### **Step 1: Data Exploration** 

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Nội dung:**
- Load và inspect dữ liệu
- Thống kê mô tả chi tiết
- Phân tích missing values
- Visualization: histograms, box plots, bar charts
- Ma trận correlation
- Phân tích các câu hỏi nghiệp vụ

**Output:** Insights về dữ liệu và biểu đồ trực quan

#### **Step 2: Data Preprocessing** 

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**Nội dung:**
- Xử lý missing values
- Loại bỏ outliers
- Feature engineering (one-hot encoding)
- Normalization (Min-Max scaling)
- Save processed data

**Output:** `data/processed/processed_data.csv`

#### **Step 3: Model Training & Evaluation** 

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

**Nội dung:**
- Load processed data
- Split train/test (80/20)
- Train Linear Regression model
- Evaluate performance
- K-Fold Cross Validation
- Feature importance analysis

**Output:** Trained model và evaluation metrics

---

## Results

### Kết quả đạt được (Metrics)

#### **Training Performance**

| Metric | Train Set | Test Set |
|--------|-----------|----------|
| **MSE** | 1,850.24 | 1,923.47 |
| **RMSE** | $43.01 | $43.86 |
| **MAE** | $29.84 | $30.12 |
| **R² Score** | 0.562 | 0.549 |

**Giải thích:**
- **RMSE ~$44:** Sai số trung bình khoảng $44, chấp nhận được
- **R² ~0.55:** Model giải thích được 55% variance
- **Train vs Test gần nhau:** Không bị overfitting

### Key Insights từ Model

**Feature Importance:**
- **Room type Entire home/apt** có ảnh hưởng mạnh nhất (+$52)
- **Manhattan location** premium cao (+$38)
- **Brooklyn** cũng có premium đáng kể (+$24)

---

## Project Structure

```
CSC17104-NUMPY-FOR-DATA-SCIENCE/
│
├── data/
│   ├── raw/
│   │   └── AB_NYC_2019.csv          # Dataset gốc
│   └── processed/
│       └── processed_data.csv       # Dữ liệu đã xử lý
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA và visualization
│   ├── 02_preprocessing.ipynb       # Data cleaning
│   └── 03_modeling.ipynb            # Training & evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Xử lý dữ liệu
│   ├── visualization.py             # Vẽ biểu đồ
│   └── models.py                    # Linear Regression
│
├── requirements.txt
└── README.md
```

### Giải thích chức năng

**src/modules:**
- `data_processing.py`: Load, clean, transform data
- `visualization.py`: Plotting functions
- `models.py`: Linear Regression implementation

---

## Challenges & Solutions

### Khó khăn với NumPy

**1. Mixed data types:**
```python
# Giải pháp: Dùng dtype=object
data = np.array(data_list, dtype=object)
```

**2. One-hot encoding:**
```python
# Implement manually
encoded = np.zeros((len(data), len(unique_values)))
```

**3. Gradient Descent không converge:**
```python
# Giải pháp: Feature scaling + tune learning rate
X = (X - X.min()) / (X.max() - X.min())
```

---

## Future Improvements

- [ ] Polynomial Regression
- [ ] Advanced feature engineering
- [ ] Web application deployment
- [ ] Time series analysis
- [ ] Interactive dashboard

---

## 👥 Contributors

**Ngô Hồng Thanh**
- Student ID: 23127475

**Course:** CSC17104 - Data Science, HK7 (2023-2024)

---

## Contact

- 📧 Email: nhthanh23@clc.fitus.edu.vn
- 🔗 GitHub: [@hnaht277](https://github.com/hnaht277)
- 📦 Repository: [CSC17104-NUMPY-FOR-DATA-SCIENCE](https://github.com/hnaht277/CSC17104-NUMPY-FOR-DATA-SCIENCE)

---

## License

This project is created for learning purposes