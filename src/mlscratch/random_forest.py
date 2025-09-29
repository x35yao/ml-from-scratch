from mlscratch.decision_tree_regressor import DecisionTreeRegressor
from mlscratch.decision_tree_classifier import DecisionTreeClassifier
from mlscratch.utils import ensure_2D
import numpy as np

class RandomForestBase:
    def __init__(self, tree_class, n_estimators, max_features = 'auto', max_depth = None, min_samples_split = 2,
                 min_samples_leaf = 1, bootstrap = True, oob_score = True, random_state = None):
        
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        
        self.tree_class = tree_class
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.trees_ = None

    
    # --- hook for subclasses to inject extra per-tree kwargs (e.g., criterion) ---
    def _extra_tree_params(self):
        return {}
    
    def _make_tree(self, seed):
        params = dict(max_depth = self.max_depth, min_samples_split = self.min_samples_split,
                      min_samples_leaf = self.min_samples_leaf, max_features = self.max_features, random_state = seed)
        params.update(self._extra_tree_params())
        return self.tree_class(**params)

    def _init_trees(self):
        seeds = self._rng.integers(0, 2**32 - 1, size=self.n_estimators)
        self.trees_ = [self._make_tree(seed) for seed in seeds]

        
    def _bootstrap(self, N):
        idx = self._rng.integers(0, N, size = N)
        oob_mask = np.ones(N, dtype = bool)
        oob_mask[idx] = False
        return idx, oob_mask
    
    def _compute_oob_score(self, X, y, oob_preds):
        self.oob_score_ = np.nan  # default no-op, will be overridden by forest classifier

    def fit(self, X, y):
        X = ensure_2D(X)
        y = np.asarray(y).ravel()
        N = len(y)

        oob_preds = None
        if self.oob_score:
            # collect per-sample OOB predictions (list per index)
            oob_preds = [[] for _ in range(N)]
        self.oob_counts_ = np.zeros(N, dtype=int)
        self._init_trees()
        for tree in self.trees_:
            if self.bootstrap:
                idx, oob_mask = self._bootstrap(N)
                tree.fit(X[idx], y[idx])
                if self.oob_score:
                    oob_idx = np.where(oob_mask)[0]
                    self.oob_counts_[oob_idx] += 1
                    if oob_idx.size:
                        yhat = tree.predict(X[oob_idx])
                        for i, pred in zip(oob_idx, yhat):
                            oob_preds[i].append(pred)
            else:
                tree.fit(X, y)

        if self.oob_score:
            self._compute_oob_score(X, y, oob_preds)
            self.oob_coverage_ = np.mean(self.oob_counts_ > 0)
        
        return self

class RandomForestClassifier(RandomForestBase):
    def __init__(self, n_estimators, criterion = 'gini', max_features='sqrt', **kwargs):
        super().__init__(tree_class=DecisionTreeClassifier,  
                         n_estimators=n_estimators,
                         max_features = max_features, **kwargs)
        self.criterion = criterion
    def _extra_tree_params(self):
        return {'criterion': self.criterion}

    def _most_frequent(self, col):
        vals, counts = np.unique(col, return_counts = True)
        winners = vals[counts == max(counts)]
        if winners.size == 1:
            return winners[0]
        return self._rng.choice(winners)
    
    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        return super().fit(X, y)
    
    def predict(self, X):
        if self.trees_ is None:
            raise ValueError("Model not trained")
        
        X = ensure_2D(X)
        # shape (n_estimators, N)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        # majority vote per sample (column)
        return np.apply_along_axis(self._most_frequent, axis=0, arr=all_preds)
    
    def predict_proba(self, X):
        if self.trees_ is None:
            raise ValueError("Model not trained")

        X = ensure_2D(X)
        C = len(self.classes_)
        N = X.shape[0]
        acc = np.zeros((N, C), dtype=float)

        for tree in self.trees_:
            probs = tree.predict_proba(X)          # shape (N, C_tree)
            tree_classes = getattr(tree, "classes_", None)
            if tree_classes is None:
                # assume same order as self.classes_ (legacy trees)
                acc += probs
            else:
                # map each tree's column into our global class order
                colmap = {c: j for j, c in enumerate(tree_classes)}
                for i, cls in enumerate(self.classes_):
                    j = colmap.get(cls, None)
                    if j is not None:
                        acc[:, i] += probs[:, j]
                    # else that tree assigned 0 prob to this class implicitly

        return acc / len(self.trees_)
    
    def _compute_oob_score(self, X, y, oob_preds):
        y = np.asanyarray(y).ravel()
        y_hat = []
        idx_used = []
        for i, preds in enumerate(oob_preds):
            if preds:  # has at least one OOB vote
                y_hat.append(self._most_frequent(np.asarray(preds)))
                idx_used.append(i)        
        if len(idx_used) == 0:
            self.oob_score_ = np.nan
        else:
            self.oob_score_ = np.mean(np.asarray(y_hat) == y[idx_used])
    
    def score(self, X, y):
        y = np.asarray(y).ravel()
        y_hat = self.predict(X)
        return np.mean(y == y_hat)


class RandomForestRegressor(RandomForestBase):
    def __init__(self, n_estimators, criterion = 'mse', max_features='auto', **kwargs):
        super().__init__(tree_class = DecisionTreeRegressor, n_estimators = n_estimators, max_features = max_features, **kwargs)
        self.criterion = criterion
    
    def _extra_tree_params(self):
        return {'criterion': self.criterion}

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        if self.trees_ is None:
            raise ValueError("Model not trained")
        X = ensure_2D(X)
        all_predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(all_predictions, axis = 0)

    def _compute_oob_score(self, X, y, oob_preds):
        idx_used = []
        y_hat = []
        for i, preds in enumerate(oob_preds):
            if preds:
                y_hat.append(np.mean(preds))
                idx_used.append(i)
        if not idx_used:
            self.oob_score_ = np.nan
            return
        y_hat = np.asarray(y_hat, dtype=float)
        y_used = np.asarray(y, dtype=float)[idx_used]
        ss_res = np.sum((y_used - y_hat) ** 2)
        ss_tot = np.sum((y_used - y_used.mean()) ** 2)
        self.oob_score_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    def score(self, X, y):
        y = np.asarray(y).ravel()
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

