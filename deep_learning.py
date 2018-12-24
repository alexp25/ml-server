import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

class DeepLearning:
    def __init__(self):
        pass

    def generate_dataset(self):

        # generate random input data to train on
        self.observations = 1000

        self.xs = np.random.uniform(low=-10, high=10, size=(self.observations, 1))
        self.zs = np.random.uniform(-10, 10, (self.observations, 1))

        # from the linear model: inputs = n x k = 1000 x 2
        self.inputs = np.column_stack((self.xs, self.zs))

        print(self.inputs.shape)

        # generate targets

        noise = np.random.uniform(-1, 1, (self.observations, 1))
        self.targets = 2 * self.xs - 3 * self.zs + 5 + noise


    def test_gradient(self):

        self.generate_dataset()
        print(self.targets.shape)

        # plot the training data
        targets = self.targets.reshape(self.observations,)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(xs, zs, targets)
        # ax.set_xlabel('xs')
        # ax.set_ylabel('zs')
        # ax.set_zlabel('Targets')
        # ax.view_init(azim=100)
        # plt.show()

        targets = targets.reshape(self.observations, 1)

        # init variables
        init_range = 0.1
        weights = np.random.uniform(-init_range, init_range, size=(2,1))
        biases = np.random.uniform(-init_range, init_range, size=1)
        print(weights)
        print(biases)

        # set a learning rate
        learning_rate = 0.02

        eps = 0.01
        max_iter = 10000

        loss = None
        outputs = None

        # train the model
        for i in range(max_iter):
            outputs = np.dot(self.inputs, weights) + biases
            deltas = outputs - targets
            # loss fcn that compares the outputs (with the current weights) to the target
            loss = np.sum(deltas ** 2)/2/self.observations
            if loss < eps:
                print('eps reached at ', i)
                break

            # print(loss)

            deltas_scaled = deltas/self.observations

            # update the weights and the biases using gradient descent
            weights = weights - learning_rate * np.dot(self.inputs.T, deltas_scaled)
            biases = biases - learning_rate * np.sum(deltas_scaled)

        print(weights)
        print(biases)
        print(loss)

        plt.plot(outputs, targets)
        plt.xlabel('outputs')
        plt.ylabel('targets')
        plt.show()

    def test_tensorflow_init(self):

        self.generate_dataset()

        # save to npz file (stores n-dimensional array)
        # data -> preprocess -> save in .npz
        # later you use the .npz to build the algorithm

        np.savez('TF_intro', inputs=self.inputs, targets=self.targets)

    def test_tensorflow_solve(self):
        input_size = 2
        output_size = 1
        inputs = tf.placeholder(tf.float32, [None, input_size])
        targets = tf.placeholder(tf.float32, [None, output_size])

        weights = tf.Variable(tf.random_uniform([input_size, output_size], minval=-0.1, maxval=0.1))
        biases = tf.Variable(tf.random_uniform([output_size], minval=-0.1, maxval=0.1))

        outputs = tf.matmul(inputs, weights) + biases

        # define the objective function
        mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs)/2.
        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)

        # open the "gates" for the data
        # it's time to execute
        sess = tf.InteractiveSession()
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        training_data = np.load('TF_intro.npz')
        for e in range(100):
            _, crt_loss = sess.run([optimize, mean_loss],
                                   feed_dict={inputs: training_data['inputs'], targets: training_data['targets']})
            print(crt_loss)

        out = sess.run([outputs],
                       feed_dict={inputs: training_data['inputs']})
        plt.plot(np.squeeze(out), np.squeeze(training_data['targets']))
        plt.xlabel('outputs')
        plt.ylabel('targets')
        plt.show()



if __name__ == '__main__':
    deeplearn = DeepLearning()
    # deeplearn.test_gradient()
    # deeplearn.test_tensorflow_init()
    deeplearn.test_tensorflow_solve()




