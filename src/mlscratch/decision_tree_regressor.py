from mlscratch.utils import node_mse, node_mae, ensure_2D
import numpy as np

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, prediction = None, is_leaf = False, depth = 0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.depth = depth

class DecisionTreeRegressor:
    def __init__(self, max_depth = 5, min_samples_split = 2, min_samples_leaf = 1, criterion = 'mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        if self.criterion == 'mse':
            self.impurity_func = node_mse
        elif self.criterion == 'mae':
            self.impurity_func = node_mae
        else:
            raise ValueError("criterion needs to be either mse or mae")

    def _stopping(self, y, depth):
        if depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        return np.all(y == y[0])
    
    def _node_prediction(self, y):
        if self.criterion == 'mse':
            prediction = np.mean(y)
        elif self.criterion == 'mae':
            prediction = np.median(y)
        return prediction
    
    def _make_leaf(self, y, depth):
        prediction = self._node_prediction(y)
        return Node(prediction=prediction, depth = depth, is_leaf = True)
    
    def _threshold_candidates(self, x):
        return (x[:-1] + x[1:]) * 0.5
    
    def _best_split(self, X, y):
        impurity_parent = self.impurity_func(y)
        N, D = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        min_samples_leaf = getattr(self, "min_samples_leaf", 1)
        for d in range(D):
            x = X[:,d]
            x_unique = np.unique(x).astype(float)
            if x_unique.size <= 1:
                continue
            thresh_candidates = self._threshold_candidates(x_unique)
            for t in thresh_candidates:
                left_idx = x <= t
                right_idx = x > t

                left_count = np.sum(left_idx)
                right_count = N  - left_count

                if left_count == 0 or right_count == 0:
                    continue
                if left_count < min_samples_leaf or right_count < min_samples_leaf:
                    continue
                impurity_children = (left_count / N * self.impurity_func(y[left_idx]) +
                                     right_count / N *self.impurity_func(y[right_idx])) 
                gain = impurity_parent - impurity_children

                if gain > best_gain:
                    best_gain = gain
                    best_feature = d
                    best_threshold = t
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        if best_feature is None:
            return None
        else:
            return best_feature, best_threshold, best_left_idx, best_right_idx

    
    def _build_tree(self, X, y, depth):
        if self._stopping(y, depth):
            return self._make_leaf(y, depth)
        
        split = self._best_split(X, y)
        if split is None:
            return self._make_leaf(y, depth)
        else:
            feature, threshold, left_idx, right_idx = split
            left_node = self._build_tree(X[left_idx], y[left_idx], depth = depth + 1)
            right_node = self._build_tree(X[right_idx], y[right_idx], depth = depth + 1)
            prediction = self._node_prediction(y)
            return Node(feature = feature,
                        threshold = threshold,
                        left = left_node,
                        right = right_node,
                        prediction = prediction,
                        depth = depth)

    def fit(self, X, y):
        X = ensure_2D(X)
        y = np.asarray(y).astype(float)

        self.root = self._build_tree(X, y, depth = 0)
        return self
    
    def _traverse_row(self, x, node):
        if node.is_leaf:
            return node
        else:
            feature = node.feature
            threshold = node.threshold
            if x[feature] <= threshold:
                return self._traverse_row(x, node.left)
            else:
                return self._traverse_row(x, node.right)
            
    def predict(self, X):
        X = ensure_2D(X)
        predictions = [self._traverse_row(x, self.root).prediction for x in X]
        return np.asarray(predictions)
    
    def score(self, X, y):
        X = ensure_2D(X)
        y = np.asarray(y, dtype=float)
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


