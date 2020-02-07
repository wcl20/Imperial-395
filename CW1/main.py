import numpy as np
import math
import sys
from tree import Node, Tree

###############################################################################
# Data Handling
###############################################################################

# Load dataset from specified path
# Rows can be accessed by data[index] e.g data[1]
# Columns can be accessed by data[name] e.g data["room"]
# Return dataset as numpy structured array
def load_data(path):
    dtype = [("s1", np.float32),
             ("s2", np.float32),
             ("s3", np.float32),
             ("s4", np.float32),
             ("s5", np.float32),
             ("s6", np.float32),
             ("s7", np.float32),
             ("room", np.float32)]
    return np.loadtxt(path, dtype=dtype)

# Splits data equally into k folds
# Returns a list of splitted data
def split_data(data, k):
    return np.split(data, k)

###############################################################################
# Main Program -- K Fold Cross Validation
###############################################################################

def main(file, pruned):
    data = load_data(file)

    # Shuffle data
    np.random.shuffle(data)

    # Split data into k folds
    k = 10
    splitted_all = split_data(data, k)

    # 4 x 4 Conusion matrix
    confusion_matrix = [[0 for _ in range(4)] for _ in range(4)]

    iteration = 0

    # For each fold ...
    for i in range(k):
        # ... each fold takes turn to be the test set
        test_set = splitted_all[i]
        remaining_set = np.concatenate((splitted_all[:i] + splitted_all[i + 1:]))

        # If the prune flag is set ...
        if pruned:

            # Split the remaining set into k - 1 folds
            splitted_remaining = split_data(remaining_set, k - 1)

            # In the remaining k - 1 folds
            for j in range(k - 1):

                iteration += 1
                print("Iteration {i}-------------------------------------------------------------------------".format(i=iteration))

                # ... each fold takes turn to be the validate set
                validate_set = splitted_remaining[j]
                training_set = np.concatenate((splitted_remaining[:j] + splitted_remaining[j + 1:]))

                # Create tree from training set
                tree = Tree(data=training_set)

                # Prune tree using reduce error pruning algorithm
                tree = tree.reduced_error_pruning(validate_set)

                # Construct confusion matrix using test set
                cmatrix = tree.build_confusion_matrix(test_set)
                # Sum up confusion matrix for finding average confusion matrix
                for i in range(4):
                    for j in range(4):
                        confusion_matrix[i][j] += cmatrix[i][j]

        # Otherwise, the prune flag is not set ...
        else:
            iteration += 1
            print("Iteration {i}-------------------------------------------------------------------------".format(i=iteration))

            # Create tree from remaining set
            tree = Tree(data=remaining_set)

            # Construct confusion matrix using test set
            cmatrix = tree.build_confusion_matrix(test_set)
            # Sum up confusion matrix for finding average confusion matrix
            for i in range(4):
                for j in range(4):
                    confusion_matrix[i][j] += cmatrix[i][j]


    # Calculate average confusion matrix
    for i in range(4):
        for j in range(4):
            confusion_matrix[i][j] /= float(iteration)

    print("Done --------------------------------------------------------------------------------")
    print("Total Number of Iterations: {i}".format(i=iteration))

    # Confusion matrix
    print("Average Confusion Matrix: -----------------------------------------------------------")
    print("\t\tPredict 1 \tPredict 2 \tPredict 3 \tPredict 4")
    for i in range(4):
        line = "Actual " + str(i + 1) + "\t"
        for j in range(4):
            line += str("{0:.2f}".format(confusion_matrix[i][j])) + "\t\t"
        print(line)

    # Recall
    print("Average Recall: ---------------------------------------------------------------------")
    total_recall = 0
    for i in range(4):
        recall = float(confusion_matrix[i][i]) / sum(confusion_matrix[i])
        total_recall += recall
        print("Room " + str(i + 1) + ": " + str(recall))

    # Precision
    print("Average Precision: ------------------------------------------------------------------")
    for i in range(4):
        print("Room " + str(i + 1) + ": " + str(float(confusion_matrix[i][i] / sum(row[i] for row in confusion_matrix))))

    # F1 Score
    print ("Average F1 Score: -------------------------------------------------------------------")
    for i in range(4):
        line = "Room " + str(i + 1) + ": "
        recall = float(confusion_matrix[i][i]) / sum(confusion_matrix[i])
        precision = float(confusion_matrix[i][i]) / sum(row[i] for row in confusion_matrix)
        line += str(2 * precision * recall / (precision + recall))
        print(line)

    # Classification Rate
    cr = sum(confusion_matrix[i][i] for i in range(4)) / sum(sum(row) for row in confusion_matrix)
    print ("Average Classification Rate: {cr}".format(cr=cr))

    # Unweighted Average Recall
    print("Unweighted Average Recall: {uar}".format(uar=float(total_recall) / 4))

    # Construct tree using entire dataset
    print("Constructing Tree using entire data set----------------------------------------------")
    # If the prune flag is set ...
    if pruned:
        # Use 20% of data as validate set
        validate_set = np.concatenate((splitted_all[:2]))
        # Use 80% of data as training set
        training_set = np.concatenate((splitted_all[2:]))

        # Create tree from training set
        tree = Tree(data=training_set)
        # Prune tree using reduced error pruning
        tree = tree.reduced_error_pruning(validate_set)

        print("Maxiaml Depth (With Pruning): {d}".format(d=tree.max_depth()))
        print("Number of Nodes (With Pruning): {n}".format(n=len(tree.get_all_nodes())))

    else:
        # Otherwise, construct tree using entire dataset
        tree = Tree(data=data)
        print("Maxial Depth (Without Pruning): {d}".format(d=tree.max_depth()))
        print("Number of Nodes (Without Pruning): {n}".format(n=len(tree.get_all_nodes())))



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing File Path")
    else:
        file = sys.argv[1]

        # Provide prune option for cross validation
        pruned =  len(sys.argv) > 2 and sys.argv[2] == "-p"
        main(file, pruned)
