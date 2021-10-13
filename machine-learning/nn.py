import numpy as np


class FCLayer:
    def __init__(self, input_size, output_size, weights=None) -> None:
        self.input_size = input_size
        self.output_size = output_size
        if weights:
            assert weights.shape == (input_size, output_size)
            self.weights = weights
        else:
            self.weights = np.ones((input_size, output_size))

    def forward(self, input_vector):
        return np.matmul(self.weights, input_vector)



class Perceptron:
    def __init__(self, input_vectors, weights, targets) -> None:
        self.input_vectors = input_vectors
        self.weights = weights
        self.targets = targets

    def forward(self):
        predicted_values = np.matmul(self.input_vectors.T, self.weights)
        predictions = np.where(predicted_values >= 0, 1, 0)
        pred_target_pairs = zip(predictions, self.targets)
        for i, (prediction, target) in enumerate(pred_target_pairs):
            if prediction != target:
                self.weights -= predicted_values[i]