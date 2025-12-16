import numpy as np


class LinearRegression:
    """
    Linear Regression model được implement từ đầu sử dụng NumPy
    
    Model: y = X @ w + b
    Loss function: MSE = (1/n) * sum((y_pred - y_true)^2)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_=0.01):
        """
        Khởi tạo Linear Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate cho gradient descent
        n_iterations : int
            Số lần lặp training
        regularization : str, optional
            Loại regularization: None, 'l1', 'l2'
        lambda_ : float
            Hệ số regularization
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Train model sử dụng gradient descent
        
        Parameters:
        -----------
        X : numpy.ndarray
            Ma trận features (n_samples, n_features)
        y : numpy.ndarray
            Vector target (n_samples,)
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            y_pred = self.predict(X)
            
            loss = self._calculate_loss(y, y_pred)
            self.loss_history.append(loss)
            
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            if self.regularization == 'l2':
                dw += (self.lambda_ / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_ / n_samples) * np.sign(self.weights)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Dự đoán giá trị output
        
        Parameters:
        -----------
        X : numpy.ndarray
            Ma trận features
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Vector dự đoán
        """
        return X @ self.weights + self.bias
    
    def _calculate_loss(self, y_true, y_pred):
        """
        Tính loss function (MSE + regularization)
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            Giá trị thực tế
        y_pred : numpy.ndarray
            Giá trị dự đoán
            
        Returns:
        --------
        loss : float
            Giá trị loss
        """
        n_samples = len(y_true)
        mse = (1 / n_samples) * np.sum((y_pred - y_true) ** 2)
        
        if self.regularization == 'l2':
            mse += (self.lambda_ / (2 * n_samples)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            mse += (self.lambda_ / n_samples) * np.sum(np.abs(self.weights))
        
        return mse
    
    def get_coefficients(self):
        """
        Lấy các hệ số của model
        
        Returns:
        --------
        dict : Dictionary chứa weights và bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias
        }


def mean_squared_error(y_true, y_pred):
    """
    Tính Mean Squared Error
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
        
    Returns:
    --------
    mse : float
        Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Tính Root Mean Squared Error
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
        
    Returns:
    --------
    rmse : float
        Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Tính Mean Absolute Error
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
        
    Returns:
    --------
    mae : float
        Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Tính R-squared (coefficient of determination)
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
        
    Returns:
    --------
    r2 : float
        R-squared score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2


def evaluate_model(y_true, y_pred):
    """
    Đánh giá model với nhiều metrics
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Giá trị thực tế
    y_pred : numpy.ndarray
        Giá trị dự đoán
        
    Returns:
    --------
    metrics : dict
        Dictionary chứa các metrics
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    return metrics


def cross_validation_split(X, y, k_folds=5):
    """
    Chia dữ liệu cho k-fold cross validation
    
    Parameters:
    -----------
    X : numpy.ndarray
        Ma trận features
    y : numpy.ndarray
        Vector target
    k_folds : int
        Số fold
        
    Returns:
    --------
    folds : list
        List các tuple (X_train, y_train, X_val, y_val)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_size = n_samples // k_folds
    folds = []
    
    for i in range(k_folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k_folds - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        folds.append((X_train, y_train, X_val, y_val))
    
    return folds


def k_fold_cross_validation(X, y, k_folds=5, learning_rate=0.01, n_iterations=1000):
    """
    Thực hiện k-fold cross validation
    
    Parameters:
    -----------
    X : numpy.ndarray
        Ma trận features
    y : numpy.ndarray
        Vector target
    k_folds : int
        Số fold
    learning_rate : float
        Learning rate
    n_iterations : int
        Số iterations
        
    Returns:
    --------
    scores : dict
        Dictionary chứa mean và std của các metrics
    """
    folds = cross_validation_split(X, y, k_folds)
    
    fold_scores = {
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R2': []
    }
    
    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(folds):
        model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        metrics = evaluate_model(y_val, y_pred)
        
        for metric_name, metric_value in metrics.items():
            fold_scores[metric_name].append(metric_value)
    
    cv_scores = {}
    for metric_name, values in fold_scores.items():
        cv_scores[f'{metric_name}_mean'] = np.mean(values)
        cv_scores[f'{metric_name}_std'] = np.std(values)
    
    return cv_scores
