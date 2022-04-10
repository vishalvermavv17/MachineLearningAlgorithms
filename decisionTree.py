import numpy as np
import pandas as pd


class DecisionTree:
    """
    Decision Tree using binary classifier.
        Parameters
        ----------
        annotate : bool
            describes the prediction path when set to True, default=False
        max_depth : int
            max permissible depth of decision tree, default=10

        Attributes
        __________
        tree : trained decision tree model.
    """

    def __init__(self, annotate=False, max_depth=10):
        self.annotate = annotate
        self.max_depth = max_depth
        self.tree = None

    def decision_tree_create(self, data, features, target, current_depth=0):
        remaining_features = features[:]  # Make a copy of the features.

        target_values = data[target]
        print("--------------------------------------------------------------------")
        print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

        # Stopping condition 1
        # (Check if there are mistakes at current node.
        # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
        if intermediate_node_num_mistakes(target_values) == 0:
            print("Stopping condition 1 reached.")
            # If not mistakes at current node, make current node a leaf node
            return create_leaf(target_values)

        # Stopping condition 2 (check if there are remaining features to consider splitting on)
        if remaining_features.empty == True:
            print("Stopping condition 2 reached.")
            # If there are no remaining features to consider, make current node a leaf node
            return create_leaf(target_values)

            # Additional stopping condition (limit tree depth)
        if current_depth >= self.max_depth:
            print("Reached maximum depth. Stopping for now.")
            # If the max tree depth has been reached, make current node a leaf node
            return create_leaf(target_values)

        # Find the best splitting feature (recall the function best_splitting_feature implemented above)
        splitting_feature = best_splitting_feature(data, features, target)

        # Split on the best feature that we found.
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
        remaining_features.drop(splitting_feature)
        print("Split on feature %s. (%s, %s)" % ( \
            splitting_feature, len(left_split), len(right_split)))

        # Create a leaf node if the split is "perfect"
        if len(left_split) == len(data):
            print("Creating leaf node.")
            return create_leaf(left_split[target])
        if len(right_split) == len(data):
            print("Creating leaf node.")
            return create_leaf(right_split[target])

        # Repeat (recurse) on left and right subtrees
        left_tree = self.decision_tree_create(left_split, remaining_features, target, current_depth + 1)
        right_tree = self.decision_tree_create(right_split, remaining_features, target, current_depth + 1)

        return {'is_leaf': False,
                'prediction': None,
                'splitting_feature': splitting_feature,
                'left': left_tree,
                'right': right_tree}

    def fit(self, x, y):
        """Fit the training data.

                Parameters
                ----------
                x : pandas dataframe, shape = [n_samples, n_features]
                    Training samples
                y : pandas sample, shape = [n_samples, 1]
                    Target values

                Returns
                -------
                self : object
                """
        features = x.columns
        target = y.name
        data = pd.concat([x, y], axis=1)
        self.tree = self.decision_tree_create(data, features, target, current_depth=0)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
                Parameters
                ----------
                x : array-like, shape = [n_samples, n_features]
                    Test samples
                Returns
                -------
                Predicted value
        """
        return x.apply(lambda row: classify(self.tree, row, self.annotate), axis=1)


def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    # Count the number of 1's (safe loans)
    safe_loans_count = sum(labels_in_node.iloc[:] == 1)

    # Count the number of -1's (risky loans)
    risky_loans_count = sum(labels_in_node.iloc[:] == -1)

    # Return the number of mistakes that the majority classifier makes.
    return min(safe_loans_count, risky_loans_count)


def best_splitting_feature(data, features, target):
    best_feature = None  # Keep track of the best feature
    best_error = 10  # Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]

        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        left_mistakes = intermediate_node_num_mistakes(left_split[target])

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])

        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error
            best_feature = feature

    return best_feature  # Return the best feature we found


def create_leaf(target_values):
    # Create a leaf node
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}

    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    # Return the leaf node
    return leaf


def classify(tree, x, annotate=False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

