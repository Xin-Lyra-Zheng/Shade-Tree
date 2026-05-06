# ADTree.py (Revised, unchanged logic; pairs with MCTS fix)
# This version implements a new tree structure where ShapeNode replaces EmptyNode.
# Any ShapeNode can act as a growth point for one or more SplitNodes.
# The prediction for a sample is the sum of scores from all reachable *terminal* ShapeNodes.

import numpy as np
from graphviz import Digraph
import logging

# Assuming the existence of a shape model class from shape_fitter.py
# This is a placeholder to make the file self-contained for understanding.
class _PlaceholderShapeModel:
    def __init__(self, constant=0.0):
        self.constant = constant
    def predict(self, X):
        return np.full(X.shape[0], self.constant)

logger = logging.getLogger('MCTS_ADTree')

# --- Node Class Hierarchy ---

class ADTreeNode:
    """Base class for all nodes in the ADTree variant."""
    def __init__(self, parent, depth, node_id):
        self.id = node_id
        self.parent = parent
        self.depth = depth

class ShapeNode(ADTreeNode):
    """
    Represents a node containing a trained feature shape function (e.g., a spline or constant).
    This node can be either terminal (a leaf) or intermediate.
    If it's intermediate, it serves as a branching point for one or more SplitNodes.
    """
    def __init__(self, parent, depth, node_id, feature_idx, feature_name, shape_model):
        super().__init__(parent, depth, node_id)
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.shape_model = shape_model
        self.children = []  # Holds SplitNode children

    def is_terminal(self):
        """Checks if the node is a leaf in the tree."""
        return not self.children

    def predict_score(self, X):
        """Predicts scores for a batch of samples using its shape model."""
        if self.shape_model:
            return self.shape_model.predict(X)
        return np.zeros(X.shape[0])

    def __repr__(self):
        status = " (Terminal)" if self.is_terminal() else f" (Branching, {len(self.children)} splits)"
        details = f"f({self.feature_name})"
        return f"ShapeNode(id='{self.id}', details='{details}{status}')"

class SplitNode(ADTreeNode):
    """
    Represents a standard decision split on a feature.
    Its parent is always a ShapeNode.
    Its children (true_child, false_child) are always new ShapeNodes.
    """
    def __init__(self, parent, depth, node_id, feature_idx, threshold, feature_name):
        super().__init__(parent, depth, node_id)
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.feature_name = feature_name
        self.true_child = None  # Will be a ShapeNode
        self.false_child = None # Will be a ShapeNode

    @property
    def cond_text(self):
        return f"{self.feature_name} <= {self.threshold:.3g}"

    def check_condition(self, X_sample):
        return X_sample[self.feature_idx] <= self.threshold

    def __repr__(self):
        return f"SplitNode(id='{self.id}', condition='{self.cond_text}')"

class ADTreeClassifier:
    """
    A classifier based on the revised Alternating Decision Tree structure.
    """
    def __init__(self, max_depth=None, min_samples_leaf=None, random_state=None, include_intermediate_shapes=False, **kwargs):
        # store config (even if not all are used yet)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.include_intermediate_shapes = bool(include_intermediate_shapes)

        self._X_train = None
        self._y_train = None
        self._feature_names = None
        self._next_node_id_counter = 0

        # The tree root is a single ShapeNode that predicts a constant value (initially 0).
        # It acts as the initial base learner in the ensemble.
        initial_model = _PlaceholderShapeModel(constant=0.0)
        self.root = ShapeNode(
            parent=None, depth=0, node_id=self._get_next_node_id(),
            feature_idx=-1, feature_name="Constant", shape_model=initial_model
        )

    def _get_next_node_id(self):
        current_id = f"node_{self._next_node_id_counter}"
        self._next_node_id_counter += 1
        return current_id

    def fit(self, X, y, feature_names=None):
        m, n = X.shape
        if feature_names is None:
            feature_names = [f"X{j}" for j in range(n)]
        self._X_train = X
        self._y_train = y
        self._feature_names = feature_names
        return self

    def _get_reachable_shape_nodes_recursive(self, current_node, x_sample, reachable_nodes, include_intermediate):
        """Recursively collect reachable shape nodes for one sample."""
        if current_node is None:
            return
        
        if isinstance(current_node, ShapeNode):
            if include_intermediate or current_node.is_terminal():
                reachable_nodes.append(current_node)
            # The sample's path continues through ALL its children.
            for split_child in current_node.children:
                self._get_reachable_shape_nodes_recursive(split_child, x_sample, reachable_nodes, include_intermediate)
        
        elif isinstance(current_node, SplitNode):
            # A split node directs the sample down one of two paths.
            if current_node.check_condition(x_sample):
                self._get_reachable_shape_nodes_recursive(current_node.true_child, x_sample, reachable_nodes, include_intermediate)
            else:
                self._get_reachable_shape_nodes_recursive(current_node.false_child, x_sample, reachable_nodes, include_intermediate)

    def decision_function(self, X):
        """Calculates the raw prediction score for each sample in X."""
        num_samples = X.shape[0]
        final_scores = np.zeros(num_samples, dtype=float)

        include_intermediate = self.include_intermediate_shapes
        if include_intermediate:
            all_shape_nodes = self._find_all_shape_nodes(self.root)
        else:
            all_shape_nodes = self._find_all_terminal_shape_nodes(self.root)
        if not all_shape_nodes:
            return final_scores

        node_predictions = {node.id: node.predict_score(X) for node in all_shape_nodes}

        # For each sample, find its reachable shape nodes and sum their pre-computed scores.
        for i in range(num_samples):
            x_sample = X[i, :]
            reachable_nodes = []
            self._get_reachable_shape_nodes_recursive(self.root, x_sample, reachable_nodes, include_intermediate)
            
            sample_score = sum(node_predictions[node.id][i] for node in reachable_nodes)
            final_scores[i] = sample_score

        return final_scores

    def _find_all_shape_nodes(self, node):
        """Helper to get all shape nodes (terminal and intermediate)."""
        if node is None:
            return []
        nodes = []
        if isinstance(node, ShapeNode):
            nodes.append(node)
            for child in node.children:
                nodes.extend(self._find_all_shape_nodes(child))
        elif isinstance(node, SplitNode):
            nodes.extend(self._find_all_shape_nodes(node.true_child))
            nodes.extend(self._find_all_shape_nodes(node.false_child))
        return nodes

    def _find_all_terminal_shape_nodes(self, node):
        """Helper to get terminal (leaf) shape nodes only."""
        if node is None:
            return []
        nodes = []
        if isinstance(node, ShapeNode):
            if node.is_terminal():
                nodes.append(node)
            else:
                for child in node.children:
                    nodes.extend(self._find_all_terminal_shape_nodes(child))
        elif isinstance(node, SplitNode):
            nodes.extend(self._find_all_terminal_shape_nodes(node.true_child))
            nodes.extend(self._find_all_terminal_shape_nodes(node.false_child))
        return nodes

    def predict(self, X):
        """Predicts the class label (-1 or 1)."""
        scores = self.decision_function(X)
        return np.where(scores > 0.0, 1, -1)

    def predict_proba(self, X):
        """Predicts class probabilities using the logistic function."""
        scores = self.decision_function(X)
        scores = np.clip(scores, -20, 20)
        prob_pos = 1 / (1 + np.exp(-scores))
        prob_neg = 1 - prob_pos
        return np.vstack([prob_neg, prob_pos]).T

# --- Visualization Functions ---

def _add_nodes_to_graph(dot, node):
    """A recursive helper to add nodes and edges to the graphviz object."""
    if node is None:
        return

    if isinstance(node, ShapeNode):
        label = f"f({node.feature_name})\nid:{node.id}"
        fill_color = 'lightgreen' if node.is_terminal() else 'lightblue'
        dot.node(str(node.id), label, shape='hexagon', style='filled', fillcolor=fill_color)
        
        # Draw edges from this ShapeNode to all its SplitNode children
        for child_split in node.children:
            dot.edge(str(node.id), str(child_split.id))
            _add_nodes_to_graph(dot, child_split)

    elif isinstance(node, SplitNode):
        label = f"{node.cond_text}\nid:{node.id}"
        dot.node(str(node.id), label, shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Draw edges from this SplitNode to its two ShapeNode children
        dot.edge(str(node.id), str(node.true_child.id), label='True')
        _add_nodes_to_graph(dot, node.true_child)
        dot.edge(str(node.id), str(node.false_child.id), label='False')
        _add_nodes_to_graph(dot, node.false_child)

def plot_ad_tree(model, filename="adtree_variant_structure", format='png'):
    """Generates and saves a visualization of the ADT variant structure."""
    dot = Digraph(name='ADTree-Variant', format=format, strict=True)
    dot.attr(rankdir='TB', splines='ortho', ranksep='0.5', nodesep='0.4')
    if filename:
        dot.graph_attr.update(label=f"ADT-Variant Structure", labelloc='t', fontsize='20')

    if model.root:
        _add_nodes_to_graph(dot, model.root)
    else:
        logger.warning("Tree is empty, cannot generate plot.")
        return None

    try:
        output_path = dot.render(filename, cleanup=True, view=False)
        logger.info(f"ADTree visualization saved to {output_path}")
        return dot
    except Exception as e:
        logger.error(f"Failed to render graphviz plot. Make sure graphviz is installed. Error: {e}")
        return None
