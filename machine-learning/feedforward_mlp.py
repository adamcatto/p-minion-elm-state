import numpy as np

def train_mlp(input_data, labels, n_epochs, bias=0.8, random_state=42, weights=None):
    """
    this is a manual, numpy implementation of the training process for a
    fully-connected network

    written for class homework: deep learning for genomics, CUNY
    Graduate Center Fall 2021
    ============================================================================

    input_data :: 2D np.ndarray, vertically-stacked input vectors

    labels :: 1D np.ndarray, ground-truth labels for error calculation

    n_epochs :: int > 0
    """
    # initialize weight matrices
    if weights is None:
        weights = []
        if random_state == 0:
            weights.append(np.zeros((3,2)))
            weights.append(np.zeros((2,1)))
        elif random_state == 1:
            weights.append(np.ones((3,2)))
            weights.append(np.ones((2,1)))
        elif random_state == 42:
            weights.append(np.array([
                [1.1,0.5],
                [0.1, -0.5],
                [1, 2]
            ]))
            weights.append(np.array([
                [0.2],
                [0.4]
            ]))

        else:
            weights.append(np.random.rand(3, 2))
            weights.append(np.random.rand(2, 1))

    else:
        assert isinstance(weights, np.ndarray)

    # add bias to each weight matrix
    for i, w in enumerate(weights):

        weights[i] = np.vstack((
            w,
            np.array([bias] * weights[i].shape[1]).reshape(1,-1)
        ))

    for j, w in enumerate(weights):
        print('initial weight matrix layer ' + str(j + 1) + ': ')
        print(w)
        print('--------')

    # construct sigmoid lambda function
    sigmoid_function = lambda z: 1 / (1 + (np.e ** (- z)))

    # initialize loss and data array for caching intermediate computations
    loss = None
    data = [input_data]

    for i in range(n_epochs):
        print('========\n========')
        print('epoch ' + str(i + 1))
        print('========\n========')

        # forward pass
        print('........\nforward pass\n........')
        inputs = input_data
        for idx, weight_matrix in enumerate(weights):
            inputs = np.hstack((
                inputs,
                np.ones((len(inputs), 1))
            ))
            inputs = np.matmul(inputs, weight_matrix)
            inputs = sigmoid_function(inputs)
            print('output of layer ' + str(idx) + ': ')
            print(inputs)

            # data.append(np.hstack((inputs, np.ones((len(inputs), 1)))))
            data.append(inputs)

        predictions = inputs

        print('predictions: \t' + str(predictions.ravel()))
        print('ground-truth labels: ' + str(labels))

        loss = np.dot((predictions.ravel() - labels), (predictions.ravel() - labels)) / 2
        print('halved mean-squared error: ' + str(loss))

        # backpropagate
        print('........\nbackward pass\n........')
        errors = [loss]
        grads = []
        reverse_data = list(reversed(data))
        for j, datum in enumerate(reverse_data[0:-1]):
            print('layer -' + str(j + 1))
            if j == 0:
                prev_datum = np.hstack((reverse_data[j + 1],
                                        np.ones((len(reverse_data[j + 1]), 1))))
                grad = np.matmul(prev_datum.T,
                                 (
                                     (labels.reshape(-1,1) - datum) *
                                        (datum * (1 - datum))))
                grad /= len(labels)
                print('gradient matrix for backwards layer ' + str(j) + ': ')
                print(grad)
                grads.append(grad)
            else:
                prev_datum = np.hstack((reverse_data[j + 1],
                                        np.ones((len(reverse_data[j + 1]), 1))))
                grad =  np.matmul(
                    prev_datum.T,
                    grads[j - 1] * datum * (1 - datum)
                )
                grad /= len(datum)
                print('gradient matrix for backwards layer ' + str(j) + ': ')
                print(grad)
                grads.append(grad)

        for j, (w, g) in enumerate(list(zip(weights, list(reversed(grads))))):
            weights[j] = weights[j] + g

        print('new weights: ')
        for j, w in enumerate(weights):
            print(j)
            print(w)
            print('========')
        data = [input_data]
