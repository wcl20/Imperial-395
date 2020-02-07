import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
from sklearn.model_selection import GridSearchCV
from nn_lib import Preprocessor
from keras_model import create_regressor, r2_score



################################################################################
# Q2.1: Implement the forward model of the ABB IRB 120
################################################################################
def main():
    """
    1) Implement a simple architecture for the regression task.
    2) Evaluates the model using evaluate_architecture() function.
    """
    # load data
    x_train, y_train, x_test, y_test= load_data("FM_dataset.dat")

    # Construct model
    model = Sequential()
    model.add(Dense(units=200, activation="relu", input_dim=3))
    model.add(Dense(units=3, activation='linear'))
    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[r2_score, "mae"])

    # Train model
    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)

    # Evalute model
    evaluate_architecture(model, x_test, y_test)

def load_data(filepath):
    """
    Loads data from given file path. Then performs shuffling, preprocessing
    and splits data into training set and test setself.
    Returns tuple (x_train, y_train, x_test, y_test)
    """
    # load data
    dat = np.loadtxt(filepath)

    # Shuffle data
    np.random.shuffle(dat)
    x = dat[:, :3]
    y = dat[:, 3:]

    # Preprocess data
    prep_input = Preprocessor(dat)
    x = prep_input.apply(x)

    # Split data into training and validation set
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return x_train, y_train, x_test, y_test

################################################################################
# Q2.2: Evaluate your architecture
################################################################################
def evaluate_architecture(model, x_test_pre, y_test):
    """
    Evaluates a model using preprocessed test set
    Returns indicators (R2) about the performance of your network.
    """
    loss, r2, mae = model.evaluate(x_test_pre, y_test, batch_size=10)
    print("Mean squared error regression loss: {}".format(loss))
    print("R2 Score: {}".format(r2))
    print("Mean Absolute Error: {}".format(mae))
    return r2, mae


################################################################################
# Q2.3: Fine tune your architecture
################################################################################
def main2():
    """
    Q2.3: Fine tune your architecture
    1) Perform hyperparameters search to search for best parameters.
    2) Use best parameters to construct a final model
    3) Perform evaluation on the final model
    4) Save model
    """
    # load preprocessed data
    x_train, y_train, x_test, y_test= load_data("FM_dataset.dat")

    # Search for best hyperparameters
    params = parameter_search(x_train, y_train)
    print("Best parameters: {p}".format(p=params))
    # Best parameters: {'activation': 'sigmoid', 'batch_size': 50, 'epochs': 300, 'neurons': 500}


    # Create a model using best params
    model = create_regressor(neurons=params['neurons'], activation=params['activation'])
    model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

    # Perform final evaluation on model
    evaluate_architecture(model, x_test, y_test)

    # Save model
    save_model(model)

def parameter_search(x_train_pre, y_train):
    """
    Performs parameter search using preprocessed training set
    The grid search parameters defined in here are:
            batch size = 10, 50, 100
            epochs = 100, 200, 300
            neurons = 100, 300, 500
            activation functions = sigmoid, relu

    GridSearchCV performs cross validation K=3 on training set to search the
    set of best hyperparameters.
    Returns the best hyperparameters set
    """

    model = KerasRegressor(build_fn=create_regressor, verbose=1)

    # define the grid search parameters
    batch_size = [10, 50, 100]
    epochs = [100, 200, 300]
    neurons = [100, 300, 500]
    activation = ["sigmoid", "relu"]
    param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, activation=activation)

    # Search for best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="r2", n_jobs=-1)
    grid_result = grid.fit(x_train_pre, y_train)

    return grid_result.best_params_

def save_model(model):
    """
    Save model to learn_FM/learn_FM.json file
    """
    with open("learn_FM/learn_FM.json", "w") as file:
        file.write(model.to_json())
    model.save_weights("learn_FM/learn_FM.h5")

################################################################################
# Q2.3: Fine tune your architecture -- Predict Hidden
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
    Load model from learn_FM/learn_FM.json file.
    Returns the compiled model.
    """
    # Load json from learn_FM/learn_FM.json
    json_file = open("learn_FM/learn_FM.json", "r")
    model = json_file.read()
    json_file.close()

    # Compile model
    model = model_from_json(model)
    model.load_weights("learn_FM/learn_FM.h5")
    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[r2_score, "mae"])
    return model

if __name__ == "__main__":
    main2()
