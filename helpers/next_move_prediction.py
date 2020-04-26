import logging
import os
import random

import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils

from .move_detection import MovesEnum

logging.getLogger("tensorflow").setLevel(logging.ERROR)
base_path = os.getcwd()


class NextMovePredictor:
    INPUT_SHAPE = (1, -1, 1)
    OUTPUT_SHAPE = (1, -1, 3)

    def __init__(self):
        self.dataset_path = os.path.join(base_path, "")
        self.model_path = os.path.join(base_path, "data", "move_predictor", "model.h5")
        self.played_moves = []
        self.model = self._create_model()
        self.load_model()

    @staticmethod
    def _create_model():
        model = Sequential()
        model.add(
            LSTM(
                units=64,
                input_shape=(None, 1),
                return_sequences=True,
                activation="sigmoid",
            )
        )
        model.add(LSTM(units=64, return_sequences=True, activation="sigmoid"))
        model.add(LSTM(units=64, return_sequences=True, activation="sigmoid"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy", "categorical_crossentropy"],
        )
        return model

    def _get_input_data(self, moves):
        # reshape data to fit model input shape
        return np.array(moves).reshape(self.INPUT_SHAPE)

    def _get_output_data(self):
        # reshape data to fit model output shape
        return np_utils.to_categorical(
            np.array(self.played_moves[1:]), num_classes=3
        ).reshape(self.OUTPUT_SHAPE)

    def train(self, user_move, verbose=0):
        # Append the new user move to the list of moves
        self.played_moves.append(user_move)
        # If we don't have at least 2 moves in the list we can't train the model
        if len(self.played_moves) <= 1:
            return
        # Format input data
        input_data = self._get_input_data(self.played_moves[:-1])
        # Format output data
        output_data = self._get_output_data()
        # Train the model
        self.model.fit(input_data, output_data, epochs=1, verbose=verbose)

    def load_model(self):
        # Load the model file if exists
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)

    def save_model(self):
        # Save the model
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        self.model.save(self.model_path)

    def predict_next_move(self):
        # If move list is empty randomly choose one move
        if not self.played_moves:
            return random.choice(list(map(int, MovesEnum.__iter__())))
        # Predict the next move
        predictions = self.model.predict(self._get_input_data(self.played_moves))
        # Get the most probable following move
        return np.argmax(predictions[0], axis=1)[0]
