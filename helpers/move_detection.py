import logging
import os
from enum import Enum

from imageai.Prediction.Custom import CustomImagePrediction

# Show only errors in console
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class MovesEnum(int, Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class ModelTypeEnum(Enum):
    """
    An helper enum to help for model type choice
    """

    RESNET = 0
    SQEEZENET = 1
    INCEPTIONV3 = 2
    DENSENET = 3


class RockPaperScissorsPredictor:
    """
    This class contains the required code for model training and move prediction using a
    webcam
    """

    MODEL_TYPE_SET_LOOKUP = {
        ModelTypeEnum.RESNET: lambda x: x.setModelTypeAsResNet(),
        ModelTypeEnum.SQEEZENET: lambda x: x.setModelTypeAsSqueezeNet(),
        ModelTypeEnum.INCEPTIONV3: lambda x: x.setModelTypeAsInceptionV3(),
        ModelTypeEnum.DENSENET: lambda x: x.setModelTypeAsDenseNet(),
    }

    MOVES_LOOKUP = {
        "rock": MovesEnum.ROCK,
        "paper": MovesEnum.PAPER,
        "scissors": MovesEnum.SCISSORS,
    }

    def __init__(
            self,
            model_type=ModelTypeEnum.RESNET,
            class_number=3,  # We have 3 different objects: "rock", "paper", "scissors"
    ):
        self.model_type = model_type
        self.class_number = class_number
        self.base_path = os.getcwd()
        # Instantiate the CustomImagePrediction object that will predict our moves
        self.predictor = CustomImagePrediction()
        # Set the model type of the neural network (it must be the same of the training)
        self._set_proper_model_type(self.model_type)
        # Set path to the trained model file
        self.predictor.setModelPath(
            os.path.join(self.base_path, "data", "move_detector", "model.h5")
        )
        # Set path to the json file that contains our classes and their labels
        self.predictor.setJsonPath(
            os.path.join(self.base_path, "data", "move_detector", "model_class.json")
        )
        # Load the trained model and set it to use "class_number" classes
        self.predictor.loadModel(num_objects=self.class_number)

    def _set_proper_model_type(self, model_type):
        self.MODEL_TYPE_SET_LOOKUP[model_type](self.predictor)

    def detect_move_from_picture(self, picture, sensibility=90):
        predictions, probabilities = self.predictor.predictImage(
            picture, result_count=3, input_type="array"
        )
        # Get a tuple (class_predicted, probability) that contains the best
        # prediction
        best_prediction = max(
            zip(predictions, probabilities), key=lambda x: x[1]
        )
        if best_prediction[1] < sensibility:
            return

        return self.MOVES_LOOKUP[best_prediction[0]]
