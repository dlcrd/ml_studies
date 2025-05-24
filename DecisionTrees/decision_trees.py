import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None,*,value = None ):
        self.feature = feature
        self.threshold = threshold
        self.left = left        # the left tree we are pointing to
        self.right = right       # The right tree we are pointing to
        self.value = value
        
    def is_leaf_node(self):
            return self.value is not None


class DecisionTree:
    """ 
    Parameters we need:
    stopping criterias:
        maximum depth
        min of samples in a node to stop
    others:
        number of features to split
        connection to root
    
    """
    def __init__(self, min_samples_split = 2,max_depth = 100, n_features = None ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) # verifying if the n_features is right.
        self.root = self._grow_tree(X,y)

    
    def _grow_tree(self, X, y, depth = 0):
        n_sample, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Verifying if the stopping conditions are satified
        if (depth >= self.max_depth) or (n_labels == 1) or (n_sample < self.min_samples_split) :
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, left=None, right=None)  # Explicit leaf node
        
        # Chose a random index
        feat_idx =np.random.choice(n_feats, self.n_features, replace = False)
        
        
        # find the best split
        
        best_feature,best_thresh = self._best_split(X,y,feat_idx)
        
        # create child nodes
        column_best_feature = X[:,best_feature]
        left_idxs, right_idxs = self._split(column_best_feature, best_thresh)
        
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1 )
        return Node(best_feature,best_thresh,left, right)
        
        
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0] # getting the most common label. and not its value.
        return value
    
    def _best_split(self,X,y,feat_idxs):
        # Find in all the thresholds that exists what is the best one.
        ## 
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                #calculate information gain
                gain = self._information_gain(y,X_column,thr)
                              
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold
    
    def _information_gain(self,y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if (len(left_idxs) == 0) or (len(right_idxs) == 0):
            return 0
        #calculate the weighted avg.e entropy of children
        n = len(y)
        n_left,n_right = len(left_idxs),len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = e_left*(n_left/n) + e_right*(n_right/n)
        
        
        # calculate te IG
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self,y):
        hist = np.bincount(y) # create a histogram or the values of y
        ps = hist/ len(y)
        sum = 0
        for p in ps:
            if p > 0:
                sum += p * np.log2(p)
        return  -sum
        
    def predict(self,X):
           return np.array([self._traverse_tree_for(x, self.root) for x in X])
       
    def _traverse_tree_for(self, x, node):
        # Imprime o nome da classe
        # print("Classe:", node.__class__.__name__)
        
        # # Lista os métodos (filtrando os métodos especiais, que iniciam com '__')
        # metodos = [attr for attr in dir(node) if callable(getattr(node, attr)) and not attr.startswith('__')]
        # print("Métodos disponíveis:", metodos)


        if node.is_leaf_node():
           return node.value
        
        if x[node.feature]<= node.threshold:
           return self._traverse_tree_for(x, node.left)
        return self._traverse_tree_for(x, node.right)