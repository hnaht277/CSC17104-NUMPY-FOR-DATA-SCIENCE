import numpy as np
import csv


def load_data(file_path):
    """
    Load dữ liệu từ file CSV sử dụng thư viện csv và NumPy
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file CSV
        
    Returns:
    --------
    data : numpy.ndarray
        Mảng NumPy chứa dữ liệu
    headers : list
        Danh sách tên các cột
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data_list = list(reader)
    
    return np.array(data_list, dtype=object), headers


def get_numeric_columns(data, headers, exclude_cols=None):
    """
    Tìm các cột số trong dữ liệu
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    headers : list
        Danh sách tên các cột
    exclude_cols : list, optional
        Danh sách các cột cần loại trừ
        
    Returns:
    --------
    numeric_indices : list
        Danh sách index của các cột số
    numeric_headers : list
        Danh sách tên các cột số
    """
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_indices = []
    numeric_headers = []
    
    for i, header in enumerate(headers):
        if header in exclude_cols:
            continue
            
        try:
            test_vals = data[:, i]
            non_empty = test_vals[test_vals != '']
            if len(non_empty) > 0:
                np.array(non_empty, dtype=float)
                numeric_indices.append(i)
                numeric_headers.append(header)
        except (ValueError, TypeError):
            continue
    
    return numeric_indices, numeric_headers


def handle_missing_values(data, headers, strategy='mean'):
    """
    Xử lý missing values trong dữ liệu
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    headers : list
        Danh sách tên các cột
    strategy : str
        Chiến lược xử lý: 'mean', 'median', 'zero', 'drop'
        
    Returns:
    --------
    cleaned_data : numpy.ndarray
        Dữ liệu đã được xử lý
    """
    data_copy = data.copy()
    numeric_indices, _ = get_numeric_columns(data_copy, headers)
    
    for idx in numeric_indices:
        col = data_copy[:, idx]
        
        numeric_mask = np.array([val != '' and val is not None for val in col])
        numeric_vals = col[numeric_mask].astype(float)
        
        if strategy == 'mean' and len(numeric_vals) > 0:
            fill_value = np.mean(numeric_vals)
        elif strategy == 'median' and len(numeric_vals) > 0:
            fill_value = np.median(numeric_vals)
        elif strategy == 'zero':
            fill_value = 0
        else:
            fill_value = 0
        
        for i in range(len(col)):
            if col[i] == '' or col[i] is None:
                data_copy[i, idx] = str(fill_value)
    
    return data_copy


def normalize_data(data, headers, method='min-max'):
    """
    Chuẩn hóa dữ liệu số
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    headers : list
        Danh sách tên các cột
    method : str
        Phương pháp chuẩn hóa: 'min-max' hoặc 'z-score'
        
    Returns:
    --------
    normalized_data : numpy.ndarray
        Dữ liệu đã được chuẩn hóa
    """
    data_copy = data.copy()
    numeric_indices, _ = get_numeric_columns(data_copy, headers)
    
    for idx in numeric_indices:
        col = data_copy[:, idx].astype(float)
        
        if method == 'min-max':
            min_val = np.min(col)
            max_val = np.max(col)
            if max_val - min_val != 0:
                normalized = (col - min_val) / (max_val - min_val)
            else:
                normalized = col
        elif method == 'z-score':
            mean_val = np.mean(col)
            std_val = np.std(col)
            if std_val != 0:
                normalized = (col - mean_val) / std_val
            else:
                normalized = col
        else:
            normalized = col
        
        data_copy[:, idx] = normalized.astype(object)
    
    return data_copy


def calculate_statistics(data, headers):
    """
    Tính toán thống kê mô tả cho dữ liệu
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    headers : list
        Danh sách tên các cột
        
    Returns:
    --------
    stats : dict
        Dictionary chứa các thống kê cho từng cột số
    """
    numeric_indices, numeric_headers = get_numeric_columns(data, headers)
    stats = {}
    
    for idx, header in zip(numeric_indices, numeric_headers):
        col = data[:, idx]
        numeric_mask = np.array([val != '' and val is not None for val in col])
        numeric_vals = col[numeric_mask].astype(float)
        
        if len(numeric_vals) > 0:
            stats[header] = {
                'count': len(numeric_vals),
                'mean': np.mean(numeric_vals),
                'std': np.std(numeric_vals),
                'min': np.min(numeric_vals),
                'max': np.max(numeric_vals),
                'median': np.median(numeric_vals),
                'q1': np.percentile(numeric_vals, 25),
                'q3': np.percentile(numeric_vals, 75)
            }
    
    return stats


def calculate_correlation(data, headers):
    """
    Tính ma trận correlation cho các cột số
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    headers : list
        Danh sách tên các cột
        
    Returns:
    --------
    corr_matrix : numpy.ndarray
        Ma trận correlation
    numeric_headers : list
        Danh sách tên các cột số
    """
    numeric_indices, numeric_headers = get_numeric_columns(data, headers)
    
    numeric_data = []
    for idx in numeric_indices:
        col = data[:, idx]
        numeric_mask = np.array([val != '' and val is not None for val in col])
        numeric_vals = col[numeric_mask].astype(float)
        numeric_data.append(numeric_vals)
    
    if len(numeric_data) > 0:
        min_length = min(len(col) for col in numeric_data)
        numeric_data = [col[:min_length] for col in numeric_data]
        numeric_array = np.array(numeric_data)
        corr_matrix = np.corrcoef(numeric_array)
    else:
        corr_matrix = np.array([])
    
    return corr_matrix, numeric_headers


def split_data(data, train_ratio=0.8, random_seed=42):
    """
    Chia dữ liệu thành tập train và test
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    train_ratio : float
        Tỷ lệ dữ liệu train (0-1)
    random_seed : int
        Seed cho random để reproducible
        
    Returns:
    --------
    train_data : numpy.ndarray
        Dữ liệu train
    test_data : numpy.ndarray
        Dữ liệu test
    """
    np.random.seed(random_seed)
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    return train_data, test_data


def extract_features(data, headers, feature_cols, target_col):
    """
    Trích xuất features và target từ dữ liệu
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào
    headers : list
        Danh sách tên các cột
    feature_cols : list
        Danh sách tên cột features
    target_col : str
        Tên cột target
        
    Returns:
    --------
    X : numpy.ndarray
        Ma trận features
    y : numpy.ndarray
        Vector target
    """
    feature_indices = [headers.index(col) for col in feature_cols if col in headers]
    target_index = headers.index(target_col) if target_col in headers else -1
    
    X = data[:, feature_indices].astype(float)
    y = data[:, target_index].astype(float)
    
    return X, y


def save_data(data, headers, file_path):
    """
    Lưu dữ liệu ra file CSV
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu cần lưu
    headers : list
        Danh sách tên các cột
    file_path : str
        Đường dẫn file output
    """
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
