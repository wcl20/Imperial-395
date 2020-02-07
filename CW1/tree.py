import numpy as np
import math

class Tree:

    def __init__(self, root=None, data=None):
        if root is None:
            self.root = self.decision_tree_learning(data)
        else:
            self.root = root

    ############################################################################
    # Data Handling
    ############################################################################

    # Sort dataset by the specified attributes:
    # "s1", "s2", "s3", "s4", "s5", "s6", "s7", "room"
    # Return sorted dataset
    def sort_data(self, data, attribute, reverse=False):
        if reverse:
            data["room"] *= -1
            sorted_data = np.sort(data, order=[attribute, "room"])
            sorted_data["room"] *= -1
            data["room"] *= -1
            return sorted_data
        else:
            return np.sort(data, order=[attribute, "room"])

    # Split dataset according to attribute < value
    # Returns tuple (attribute < value, attribute > value)
    def split_data(self, data, attribute, value):
        data = self.sort_data(data, attribute)
        i = np.where(data[attribute] > value)[0][0]
        return data[:i], data[i:]

    ############################################################################
    # Creating Decision Tree
    ############################################################################

    # Creates a decision tree from the given dataset
    # Returns a decision tree
    def decision_tree_learning(self, data):

        if self.same_label(data):
            return Node("room", data["room"][0], data["room"][0])
        else:
            # Find best split point
            sps = self.get_all_split_points(data)
            best_sp = None
            best_gain = None
            for sp in sps:
                gain = self.calculate_gain(data, sp)
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_sp = sp

            # Find most common room in data
            most_common_room = self.get_most_common_room(data)

            # The program may fail to find a split point in the case of noisy data,
            # e.g. (51, 52, 53, 54, 55, 56, 57, 3), (51, 52, 53, 54, 55, 56, 57, 1)
            if best_sp is None:
                print("No split point")
                # Return a node with the most common room
                return Node("room", most_common_room, most_common_room)

            # Create new node for split point
            node = Node(best_sp[0], best_sp[1], most_common_room)

            # Split dataset according to best split point
            l_data, r_data = self.split_data(data, best_sp[0], best_sp[1])

            # Recursive call for remaining dataset
            node.left = self.decision_tree_learning(l_data)
            node.right = self.decision_tree_learning(r_data)

            return node

    # Finds all split points in the given dataset
    # Split points are between two examples in sorted order that have
    # different values
    # Returns a list of tuple [(attribute, value)]
    def get_all_split_points(self, data):
        split_points = set()
        for attribute in ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]:
            sorted_data = self.sort_data(data, attribute)
            for row in range(len(sorted_data) - 1):
                if sorted_data[row][attribute] != sorted_data[row + 1][attribute]:
                    split_points.add((attribute, sorted_data[row][attribute]))
        return split_points

    # Returns true if all samples have the same label
    def same_label(self, data):
        return len(np.unique(data["room"])) == 1

    # Returns the most common room in the data
    def get_most_common_room(self, data):
        room_count = self.get_room_count(data)
        return max(zip(room_count.values(), room_count.keys()))[1]

    # Count the number of each room in the dataset
    # Returns a dictionary with {Room number: count}
    def get_room_count(self, data):
        unique, counts = np.unique(data["room"], return_counts=True)
        return dict(zip(unique, counts))

    # Calculate entropy for given dataset
    # Returns entropy of the data
    def calculate_entropy(self, data):
        room_count = self.get_room_count(data)
        total = sum(room_count.values())
        entropy = 0.0
        for rooms in room_count.values():
            p = rooms / total
            entropy -= p * math.log(p, 2)
        return entropy

    # Caculate information gain by splitting dataset at sp
    def calculate_gain(self, data, sp):
        l_data, r_data = self.split_data(data, sp[0], sp[1])
        remainder = len(l_data) / len(data) * self.calculate_entropy(l_data) + \
                    len(r_data) / len(data) * self.calculate_entropy(r_data)
        return self.calculate_entropy(data) - remainder

    ############################################################################
    # Evaluation
    ############################################################################

    # Evaluate the trained tree against validate set
    # Returns the number of wrong classifications in validate set
    def evaluate(self, validate_set):
        error = 0
        for sample in validate_set:
            if not self.classify_sample(sample) == sample["room"]:
                error += 1
        return error

    # Classify a sample using trained tree
    # Sample is in the format (s1, s2, s3, s4 ,s5 ,s6 ,s6, room)
    # Returns the predicted room number
    def classify_sample(self, sample):
        return self.classify_sample_recursive(self.root, sample)

    # Helper function use to classfiy sample
    def classify_sample_recursive(self, node, sample):
        if node.attribute == "room":
            return node.value
        if sample[node.attribute] <= node.value:
            return self.classify_sample_recursive(node.left, sample)
        else:
            return self.classify_sample_recursive(node.right, sample)

    # Return the maximum depth of the tree
    def max_depth(self):
        return self.max_depth_recursive(self.root)

    # Helper Function to find the maximum depth
    def max_depth_recursive(self, root):
        if root is None:
            return 0
        else:
            # Compute the depth of each subtree
            l_depth = self.max_depth_recursive(root.left)
            r_depth = self.max_depth_recursive(root.right)

            # Use the larger one
            if l_depth > r_depth:
                return l_depth + 1
            else:
                return r_depth + 1

    # Builds a confusion matrix on the given dataset
    # Returns a 4x4 matrix in the form
    #
    def build_confusion_matrix(self, test_set):
        confusion_matrix = [[0 for _ in range(4)] for _ in range(4)]
        for sample in test_set:
            actual = int(sample["room"])
            predict = int(self.classify_sample(sample))
            confusion_matrix[predict - 1][actual - 1] += 1
        return confusion_matrix

    ############################################################################
    # Pruning
    ############################################################################

    # Reduced Error Pruning Algorithm prunes tree using validation set
    def reduced_error_pruning(self, validate_set):
        current_best_tree = self
        current_min_error = current_best_tree.evaluate(validate_set)

        while True:
            min_error = current_min_error
            best_tree = current_best_tree

            # Consider each non leaf node for pruning ...
            for node in [n for n in current_best_tree.get_all_nodes() if not n.is_leaf()]:
                pruned_tree = current_best_tree.prune_tree(node)
                new_error = pruned_tree.evaluate(validate_set)
                # ... choose node whose removal most increases the tree accuracy
                if new_error < min_error:
                    best_tree = pruned_tree
                    min_error = new_error

            # Continue pruning until pruning is harmful
            if best_tree == current_best_tree:
                break
            else:
                current_best_tree = best_tree
                current_min_error = min_error

        return current_best_tree

    # Returns a new tree with the specified target node pruned
    def prune_tree(self, target):
        root = self.prune_tree_recursive(self.root, target)
        return Tree(root)

    # Helper Function use to prune tree
    def prune_tree_recursive(self, node, target):
        if node is None:
            return node
        # If the node is the target node to be pruned ...
        if node == target:
            # ... turn target node into a leaf, and assign most common room
            return node.to_leaf()
        else:
            # Otherwise, make a copy of the node with same attributes ...
            new_node = node.copy()
            # ... and traverse down subtrees
            new_node.left = self.prune_tree_recursive(node.left, target)
            new_node.right = self.prune_tree_recursive(node.right, target)
            return new_node

    # Return a list will all nodes of the tree
    # The list is ordered starting from the lowest level to root node
    def get_all_nodes(self):
        return self.get_all_nodes_recursive(self.root)

    # Helper Function use to find all nodes in BFS manner
    def get_all_nodes_recursive(self, node):
        queue = [node]
        nodes = []
        while (not queue == []):
            n = queue.pop(0)
            nodes.append(n)
            if not n.left is None:
                queue.append(n.left)
            if not n.right is None:
                queue.append(n.right)
        return nodes[::-1]

    ############################################################################
    # Print Tree
    ############################################################################

    def __str__(self):
        return self.str_recursive(self.root, 0)

    # Helper Function use to print tree
    def str_recursive(self, root, level):
        if root is None:
            return ""
        result = str(root) + "\n"
        if not root.left is None:
            result += "\t" * (level + 1)
            result += "Left: " + self.str_recursive(root.left, level + 1)
        if not root.right is None:
            result += "\t" * (level + 1)
            result += "Right: " + self.str_recursive(root.right, level + 1)
        return result



# It is advised to store nodes as a single object,
# using this kind of structure {’attribute’, ’value’, ’left’, ’right’}
# You might also want to add a Boolean field name ”leaf” that indicates
# whether or not the node is a leaf
class Node:

    def __init__(self, attribute, value, most_common_room, left=None, right=None):
        self.attribute = attribute
        self.value = value
        self.most_common_room = most_common_room
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    # Returns a new node transformed to leaf
    # Attribute becomes room, value becomes the most common room
    def to_leaf(self):
        return Node("room", self.most_common_room, self.most_common_room)

    # Returns a copy of the node with stored attribute
    def copy(self):
        return Node(self.attribute, self.value, self.most_common_room)

    ############################################################################
    # Print Node
    ############################################################################

    def __str__(self):
        return "Node[ attribute: {attr}, value: {val} ]" \
            .format(attr=self.attribute, val=self.value)
