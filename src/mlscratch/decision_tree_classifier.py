from mlscratch.utils import ensure_2D, gini, entropy, class_probs
import numpy as np

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, 
                 probs = None, prediction = None, is_leaf = False, depth = 0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.probs = probs
        self.prediction = prediction
        self.is_leaf = is_leaf
        self.depth = depth

class DecisionTreeClassifier:
    def __init__(self, max_depth = 5, min_samples_leaf = 1, min_samples_split = 2, criterion = 'gini',
                 max_features = None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        if criterion == 'gini':
            self.criterion = gini
        elif criterion == 'entropy':
            self.criterion = entropy
        else:
            raise ValueError("criterion needs to be either gini or entropy")
        
    def _resolve_max_features(self, d):
        if not self.max_features or self.max_features in ['all', 'max']:
            return d
        if isinstance(self.max_features, str):
            if self.max_features == 'log2':
                return max(1, int(np.log2(d)))
            if self.max_features == 'sqrt' or self.max_features == 'auto':
                return max(1, int(np.sqrt(d)))
            else:
                raise ValueError('Unrecognized string for max_features')
        if isinstance(self.max_features, int):
            if self.max_features <= 0:
                raise ValueError("max_features must be a positive integer")
            return min(d, self.max_features)
        if isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("max_features must be a float in (0, 1]")
            return max(int(d * self.max_features), 1)
        raise ValueError(f"Invalid type for max_features: {type(self.max_features)}")

    def _classes_probs(self, y):
        return class_probs(y, n_classes=len(self.classes_))

    def _make_leaf(self, y, depth):
        probs = self._classes_probs(y)
        pred_idx = np.argmax(probs)
        return Node(probs = probs, prediction = self.classes_[pred_idx], depth = depth, is_leaf = True)

    def _stopping(self, y, depth):
        if depth >= self.max_depth:
            return True
        
        if len(y) < self.min_samples_split:
            return True
        # pure?
        return np.all(y == y[0])
    
    def _threshold_candidates(self, x):
        return (x[:-1] + x[1:]) * 0.5
    

    def _best_split(self, X, y):
        N, D = X.shape

        d = self._resolve_max_features(D)                 # how many features to try at this node
        feat_idx = self._rng.choice(D, size=d, replace=False)  # choose subset

        best_gain = -np.inf
        best_left_idx = None
        best_right_idx = None
        best_feature = None
        best_threshold = None

        min_samples_leaf = getattr(self, 'min_samples_leaf', 1)
        probs_parent = self._classes_probs(y)
        impurity_parent = self.criterion(probs_parent)
        for d_idx in feat_idx:
            x = X[:, d_idx]
            x_unique = np.unique(x)
            if x_unique.size <= 1: # can't split
                continue
            threshold_candidates = self._threshold_candidates(x_unique)
            for t in threshold_candidates:
                left_idx = x <= t
                right_idx = x > t

                left_count = np.sum(left_idx)
                right_count = N - left_count

                if left_count == 0 or right_count == 0:
                    continue
                if left_count < min_samples_leaf or right_count < min_samples_leaf:
                    continue

                probs_left = self._classes_probs(y[left_idx])
                probs_right = self._classes_probs(y[right_idx])

                impurity_children = (left_count / N * self.criterion(probs_left) + 
                                     right_count / N * self.criterion(probs_right))
                gain = impurity_parent - impurity_children

                if gain > best_gain:
                    best_gain = gain
                    best_feature = d_idx
                    best_threshold = t
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        if best_feature is None:
            return None
        else:
            return best_feature, best_threshold, best_left_idx, best_right_idx          

    def _build_tree(self, X, y, depth = 0):
        if self._stopping(y, depth):
            return self._make_leaf(y, depth)
        
        N = X.shape[0]
        split = self._best_split(X, y)
        if split is None:
            return self._make_leaf(y, depth)
        else:
            feature, threshold, left_idx, right_idx = split

        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[right_idx], y[right_idx]

        left_node = self._build_tree(X_left, y_left, depth = depth + 1)
        right_node = self._build_tree(X_right, y_right, depth = depth + 1)

        probs = self._classes_probs(y)
        pred_idx = np.argmax(probs)
        return Node(feature = feature,
                    threshold = threshold,
                    left = left_node, 
                    right = right_node,
                    probs = probs,
                    prediction = self.classes_[pred_idx],
                    depth = depth)
        
    def fit(self, X, y):
        self._rng = np.random.default_rng(self.random_state)
        X = ensure_2D(X)
        y = np.asarray(y)
        classes, y_idx = np.unique(y, return_inverse = True)
        self.classes_ = classes
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y_idx, depth = 0)
        return self
    
    def _traverse_row(self, x_row, node):
        if node.is_leaf:
            return node
        else:
            feature = node.feature
            threshold = node.threshold

            if x_row[feature] <= threshold:
                return self._traverse_row(x_row, node.left)
            else:
                return self._traverse_row(x_row, node.right)

    def predict(self, X):
        X = ensure_2D(X)
        predictions = [self._traverse_row(x_row, self.root).prediction for x_row in X]
        return np.asarray(predictions)
    
    def predict_proba(self, X):
        X = ensure_2D(X)
        pred_probs = [self._traverse_row(x_row, self.root).probs for x_row in X]
        return np.vstack(pred_probs)
    
    def score(self, X, y):
        X = ensure_2D(X)
        y = np.asarray(y).ravel()
        y_hat = self.predict(X)
        return np.mean(y_hat == y)


        