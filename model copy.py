from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

def build_neural_network(input_dim):
    """
    Build a neural network model for stock price prediction.
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Hidden layers with dropout
    dense_1 = Dense(256, activation="relu")(input_layer)
    dropout_1 = Dropout(0.4)(dense_1)

    dense_2 = Dense(128, activation="relu")(dropout_1)
    dropout_2 = Dropout(0.3)(dense_2)

    dense_3 = Dense(64, activation="relu")(dropout_2)
    dropout_3 = Dropout(0.2)(dense_3)

    # Output layer
    output_layer = Dense(1)(dropout_3)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model
