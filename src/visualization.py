import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(data, column_name, title=None, color='skyblue', bins=30):
    """
    Vẽ biểu đồ phân phối (histogram) cho một cột dữ liệu
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào (1D array)
    column_name : str
        Tên cột dữ liệu
    title : str, optional
        Tiêu đề biểu đồ
    color : str, optional
        Màu sắc biểu đồ
    bins : int, optional
        Số lượng bins cho histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(title if title else f'Distribution of {column_name}')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_box(data, column_name, title=None, color='lightblue'):
    """
    Vẽ box plot để hiển thị phân phối và outliers
    
    Parameters:
    -----------
    data : numpy.ndarray
        Dữ liệu đầu vào (1D array)
    column_name : str
        Tên cột dữ liệu
    title : str, optional
        Tiêu đề biểu đồ
    color : str, optional
        Màu sắc biểu đồ
    """
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, vert=True, patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor(color)
    plt.ylabel(column_name)
    plt.title(title if title else f'Box Plot of {column_name}')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scatter(x_data, y_data, x_label, y_label, title=None, color='blue', alpha=0.5):
    """
    Vẽ scatter plot để hiển thị mối quan hệ giữa 2 biến
    
    Parameters:
    -----------
    x_data : numpy.ndarray
        Dữ liệu trục X
    y_data : numpy.ndarray
        Dữ liệu trục Y
    x_label : str
        Nhãn trục X
    y_label : str
        Nhãn trục Y
    title : str, optional
        Tiêu đề biểu đồ
    color : str, optional
        Màu sắc điểm
    alpha : float, optional
        Độ trong suốt
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color=color, alpha=alpha, edgecolors='black', linewidth=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title if title else f'{y_label} vs {x_label}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(corr_matrix, labels, title='Correlation Matrix'):
    """
    Vẽ heatmap cho ma trận correlation
    
    Parameters:
    -----------
    corr_matrix : numpy.ndarray
        Ma trận correlation
    labels : list
        Danh sách tên các biến
    title : str, optional
        Tiêu đề biểu đồ
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=labels, yticklabels=labels, 
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_distributions(data_dict, title='Multiple Distributions', figsize=(15, 10)):
    """
    Vẽ nhiều biểu đồ phân phối trong một figure
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary với key là tên cột và value là dữ liệu
    title : str, optional
        Tiêu đề chung
    figsize : tuple, optional
        Kích thước figure
    """
    n_plots = len(data_dict)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_plots))
    
    for idx, (col_name, data) in enumerate(data_dict.items()):
        axes[idx].hist(data, bins=30, color=colors[idx], edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(col_name)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {col_name}')
        axes[idx].grid(axis='y', alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_comparison(data1, data2, label1, label2, title, xlabel, bins=30):
    """
    So sánh 2 phân phối dữ liệu
    
    Parameters:
    -----------
    data1 : numpy.ndarray
        Dữ liệu thứ nhất
    data2 : numpy.ndarray
        Dữ liệu thứ hai
    label1 : str
        Nhãn cho dữ liệu 1
    label2 : str
        Nhãn cho dữ liệu 2
    title : str
        Tiêu đề biểu đồ
    xlabel : str
        Nhãn trục X
    bins : int, optional
        Số lượng bins
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=bins, alpha=0.5, label=label1, color='blue', edgecolor='black')
    plt.hist(data2, bins=bins, alpha=0.5, label=label2, color='red', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_regression_results(y_true, y_pred, title='Actual vs Predicted'):
    """
    Vẽ biểu đồ so sánh giá trị thực tế và dự đoán
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
    title : str, optional
        Tiêu đề biểu đồ
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title='Residual Plot'):
    """
    Vẽ biểu đồ residuals
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
    title : str, optional
        Tiêu đề biểu đồ
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importance_values, title='Feature Importance'):
    """
    Vẽ biểu đồ tầm quan trọng của features
    
    Parameters:
    -----------
    feature_names : list
        Danh sách tên features
    importance_values : numpy.ndarray
        Giá trị tầm quan trọng
    title : str, optional
        Tiêu đề biểu đồ
    """
    sorted_idx = np.argsort(np.abs(importance_values))[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_values)), np.abs(importance_values)[sorted_idx], color='skyblue')
    plt.yticks(range(len(importance_values)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
