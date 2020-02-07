import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras_model import create_classifier
from keras.models import model_from_json

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from nn_lib import (
    Preprocessor,
    save_network,
    load_network,
)

################################################################################
# Q3.1: Implement the ROI detector
################################################################################

def main():

    # load data
    x_train_pre, y_train, x_val_pre, y_val = load_data("ROI_dataset.dat")

    #Construct model
    model = Sequential()
    model.add(Dense(units=500, activation="relu", input_dim=3))
    model.add(Dense(units=4, activation="softmax"))

    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.1, clipvalue=0.5)

    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

    # Train model
    model.fit(x_train_pre, y_train, epochs=300, batch_size=50, verbose=1)

    evaluate_architecture(model, x_val_pre, y_val)

def load_data(filepath):
    # load data
    unprocessed_dat = np.loadtxt(filepath)

    dat = rebalancing_dataset(unprocessed_dat)
    # Shuffle data
    np.random.shuffle(dat)
    x = dat[:, :3]
    y = dat[:, 3:]

    # Split data into training and validation set
    split_idx = int(0.8 * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Preprocess data
    prep_input = Preprocessor(dat)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    return x_train_pre, y_train, x_val_pre, y_val

def rebalancing_dataset(dat):
    """
    Balances the dataset but upsampling and downsampling the data provided
    """
    region1 = dat[np.where(dat[:, 3] == 1)]
    region2 = dat[np.where(dat[:, 4] == 1)]
    region3 = dat[np.where(dat[:, 5] == 1)]
    region4 = dat[np.where(dat[:, 6] == 1)]

    desired_size = int(region2.size / 7)

    region3_upsampled = region3[np.random.choice(region3.shape[0], desired_size, replace=True), :]
    region4_downsampled = region4[np.random.choice(region4.shape[0], desired_size, replace=False), :]

    balanced_set = np.concatenate((region1, region2, region3_upsampled, region4_downsampled))

    return balanced_set

################################################################################
# Q3.2: Evaluate your architecture
################################################################################

def evaluate_architecture(model, x_test_pre, y_test):
    """
    Evaluates a model using preprocessed test set
    Returns indicators (acc) about the performance of your network.
    """
    loss, acc = model.evaluate(x_test_pre, y_test, batch_size=10)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(acc))
    pred = model.predict(x_test_pre)
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
    precision, recall, fscore, support = score(y_test.argmax(axis=1), pred.argmax(axis=1))
    print(conf_matrix)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return acc

################################################################################
# Q3.3: Fine tune your architecture
################################################################################

def main2():
    """
    Q3.3: Fine tune your architecture
    1) Perform hyperparameters search to search for best parameters.
    2) Use best parameters to construct a final model
    3) Perform evaluation on the final model
    4) Save model
    """
    # load preprocessed data
    x_train_pre, y_train, x_test, y_test= load_data("ROI_dataset.dat")

    # Search for best hyperparameters
    params = parameter_search(x_train_pre, y_train)
    print("Best parameters: {p}".format(p=params))
    # Best parameters: {'activation': 'relu', 'batch_size': 50, 'epochs': 300, 'neurons':500}

    # Create a model using best params
    model = create_classifier(learn_rate=params['learn_rate'], clipvalue=params['clipvalue'])
    model.fit(x_train_pre, y_train, epochs=300, batch_size=50, verbose=0)

    # Perform final evaluation on model
    evaluate_architecture(model, x_test, y_test)

    # Save model
    save_model(model)

def parameter_search(x_train_pre, y_train):
    """
    Performs parameter search using preprocessed training set
    The grid search parameters defined in here are:
            learn_rate = [0.1, 0.2, 0.3]
            clipvalue = [0.5, 0.6, 0.7, 0.8, 0.9]

    GridSearchCV performs cross validation K=3 on training set to search the
    set of best hyperparameters.
    Returns the best hyperparameters set
    """
#    model = KerasClassifier(build_fn=create_classifier, verbose=1)
    model = KerasClassifier(build_fn=create_classifier, epochs=300, batch_size=50, verbose=1)

    # define the grid search parameters
    learn_rate = [0.1, 0.2, 0.3]
    clipvalue = [0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(learn_rate=learn_rate, clipvalue=clipvalue)

    # Search for best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_train_pre, y_train)

    return grid_result.best_params_

def save_model(model):
    """
    Save model to learn_ROI/learn_ROI.json file
    """
    with open("learn_ROI/learn_ROI.json", "w") as file:
        file.write(model.to_json())
    model.save_weights("learn_ROI/learn_ROI.h5")

################################################################################
# Q3.3: Fine tune your architecture -- Predict Hidden
################################################################################
def predict_hidden(dat):
    """
    Given a NumPy array dataset:
    1) Preprocess the given data
    2) Loads the best performing model
    3) Returns the prediction using the model
    """
    # Preprocess data
    x, y = dat[:, :3], dat[:, 3:]
    prep_input = Preprocessor(dat)
    x_pre = prep_input.apply(x)

    # Load model
    model = load_model()

    # Predict data
    return model.predict(x_pre)

def load_model():
    """
    Load model from learn_ROI/learn_ROI.json file.
    Returns the compiled model.
    """
    # Load json from learn_FM/learn_FM.json
    json_file = open("learn_ROI/learn_ROI.json", "r")
    model = json_file.read()
    json_file.close()

    # Compile model
    model = model_from_json(model)
    model.load_weights("learn_ROI/learn_ROI.h5")
    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.3, clipvalue=0.9)
    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])
    return model

if __name__ == "__main__":
    main3()
