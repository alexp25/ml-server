import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn import preprocessing

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


    # using the audiobooks data to create a model
    # that will predict if a customer will convert
    def audiobook_case_study_full(self):
        self.read_audiobook_data()

    def read_audiobook_data(self):
        raw_csv_data = np.loadtxt("./data/Audiobooks-data.csv", delimiter=",")
        # print(raw_csv_data)
        # take all columns excl the ID (col 0) and the targets (last col/-1)
        # all rows, cols 1 to last-1
        unscaled_inputs_all = raw_csv_data[:, 1:-1]
        # take the targets (last col/-1)
        # all rows, col last
        # the targets represent the output (e.g. showing that the client converted or not)
        # with this target/outcome we should go on building the model that will predict on further data
        targets_all = raw_csv_data[:, -1]
        # print(targets_all)

        # balance the dataset
        # we know that there are more records with no review that records with review!

        # number of records with review = number of records with no review
        # get number of records with review (targets=1)
        num_one_targets = int(np.sum(targets_all))
        zero_targets_counter = 0
        indices_to_remove = []

        for i in range(targets_all.shape[0]):
            if targets_all[i] == 0:
                zero_targets_counter += 1
                if zero_targets_counter > num_one_targets:
                    indices_to_remove.append(i)

        # remove additional records from the no review class
        unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
        targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

        # standardize the inputs (scale)
        # Center to the mean and component wise scale to unit variance
        scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
        # print("scaled: ", scaled_inputs)

        # shuffle the data
        shuffled_indices = np.arange(scaled_inputs.shape[0])
        np.random.shuffle(shuffled_indices)
        shuffled_inputs = scaled_inputs[shuffled_indices]
        shuffled_targets = targets_equal_priors[shuffled_indices]

        # split the dataset into train, validation and test
        samples_count = shuffled_inputs.shape[0]
        train_samples_count = int(0.8*samples_count)
        validation_samples_count = int(0.1*samples_count)
        test_samples_count = samples_count - train_samples_count - validation_samples_count

        # the first section is the train data
        train_inputs = shuffled_inputs[:train_samples_count]
        train_targets = shuffled_targets[:train_samples_count]

        # the second section is the validation data
        validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
        validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

        # the third section is the test data (on which we will test the model)
        test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
        test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

        # show the proportion of ones as a part of the total for the 3 data sets
        print("proportions")
        print("train")
        print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)
        print("validation")
        print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/validation_samples_count)
        print("test")
        print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)

        # save the 3 datasets in npz
        np.savez("Audiobooks_data_train", inputs=train_inputs, targets=train_targets)
        np.savez("Audiobooks_data_validation", inputs=validation_inputs, targets=validation_targets)
        np.savez("Audiobooks_data_test", inputs=test_inputs, targets=test_targets)

    def get_audiobooks_model(self):

        # the number of input variables (here: each column in the dataset)
        input_size = 10
        # the number of outputs (converted or not, one-hot encoding)
        output_size = 2
        # height (higher values increase the accuracy)
        hidden_layer_size = 100

        # we use 2 hidden layers (width)
        n_hidden_layers = 3

        batch_size = 100

        max_epochs = 100
        prev_validation_loss = 9999999.

        # reset the computational graph
        tf.reset_default_graph()

        inputs = tf.placeholder(tf.float32, [None, input_size])
        targets = tf.placeholder(tf.int32, [None, output_size])

        # the first hidden layer (from 10 to 50)
        weights_crt_layer = tf.get_variable("weights_1", [input_size, hidden_layer_size])
        biases_crt_layer = tf.get_variable("biases_1", [hidden_layer_size])
        # the activation function of the hidden layers is relu
        # find the nodes of the first hidden layer
        outputs_crt_layer = tf.nn.relu(tf.matmul(inputs, weights_crt_layer) + biases_crt_layer)

        # the hidden layers
        for i in range(n_hidden_layers - 1):
            # the ith hidden layer (from 50 to 50)
            print("compute hidden layer {0}".format(i+2))
            weights_crt_layer = tf.get_variable("weights_{0}".format(i + 2), [hidden_layer_size, hidden_layer_size])
            biases_crt_layer = tf.get_variable("biases_{0}".format(i + 2), [hidden_layer_size])
            # the activation function of the hidden layers is relu
            outputs_crt_layer = tf.nn.relu(tf.matmul(outputs_crt_layer, weights_crt_layer) + biases_crt_layer)
            # others are: sigmoid, tanh, relu, softmax

        # the output layer (from 50 to 2)
        weights_3 = tf.get_variable("weights_{0}".format(hidden_layer_size), [hidden_layer_size, output_size])
        biases_3 = tf.get_variable("biases_{0}".format(hidden_layer_size), [output_size])
        # will produce probabilities (e.g. 0.4, 0.6)
        outputs = tf.matmul(outputs_crt_layer, weights_3) + biases_3

        # labels = targets (supervised learning
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets)
        mean_loss = tf.reduce_mean(loss)

        # choose the optimization method
        # optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)
        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)

        # prediction accuracy
        # showing in what percentage of the cases the output of the algorithm matched the target
        # the predicted target is taken as the arg of the highest probability in the one-hot encoded outputs
        # we compare that to the one-hot encoded targets
        # should be equal if there is a match
        # e.g. [2, 1, 0, 0] vs [2, 0, 0, 0] => 3/4, 75% accuracy
        out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
        # get the mean accuracy
        accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

        sess = tf.InteractiveSession()
        # initialize the variables
        initializer = tf.global_variables_initializer()
        sess.run(initializer)

        # now load the data
        train_data = Audiobooks_Data_Reader('train', batch_size)
        validation_data =  Audiobooks_Data_Reader('validation')

        # optimization section (make it learn)
        # multiple epochs (generations)
        for epoch_counter in range(max_epochs):
            curr_epoch_loss = 0.
            # training part (in batches)
            for input_batch, target_batch in train_data:
                # we optimize the mean loss in batches
                _, batch_loss = sess.run([optimize, mean_loss],
                feed_dict={inputs: input_batch, targets: target_batch})
                curr_epoch_loss += batch_loss

            curr_epoch_loss /= train_data.batch_count

            validation_loss = 0.
            validation_accuracy = 0.

            # validation part
            # we validate the model on the validation data set now
            # --------- we use this part to prevent overfitting of weight and bias parameters ---------
            # forward propagate
            for input_batch, target_batch in validation_data:
                # we record the validation loss and the accuracy
                validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch})

            print('Epoch: ' + str(epoch_counter+1) +
                  '. Training loss: ' + '{0:.3f}'.format(curr_epoch_loss)+
                  '. Validation loss: ' + '{0:.3f}'.format(validation_loss)+
                  '. Validation accuracy: ' + '{0:.2f}'.format(validation_accuracy*100.)+'%')

            if validation_loss > prev_validation_loss:
                # e.g. the model started overfitting
                break

            prev_validation_loss = validation_loss
        print('End of training.')

        test_data = Audiobooks_Data_Reader('test')
        # test part (prediction)
        # --------- we use this part to help preventing overfitting of hyperparameters (width, depth, learning rate, etc.) ---------
        for input_batch, target_batch in test_data:
            # we record the validation loss and the accuracy
            test_accuracy = sess.run([accuracy],
            feed_dict={inputs: input_batch, targets: target_batch})

        test_accuracy_percent = test_accuracy[0]*100.
        print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')


        # test make predictions with the model
        # https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
        test_data = Audiobooks_Data_Reader('test')


        # tf_test_dataset = tf.constant(test_data)
        # tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        # feed_dict = {tf_train_dataset: batch_data}
        # test_prediction = tf.nn.softmax(model(tf_test_dataset, False))
        #
        # predictions = sess.run([test_prediction], feed_dict)
        # print(predictions)

        # y_pred = []
        # my_inputs = tf.placeholder(tf.float32, [None, input_size])
        # test_data = Audiobooks_Data_Reader('test')
        # for input_batch, target_batch in test_data:
        #     # print(input_batch)
        #     output_batch = sess.run([y_pred], feed_dict={my_inputs: input_batch})
        #     print(output_batch)

        # SAVE_PATH = './models'
        # saver = tf.train.Saver()
        # path = saver.save(sess, SAVE_PATH + '/Audiobooks.ckpt')
        # print("saved at {}".format(path))
        #
        # checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
        # graph = tf.get_default_graph()
        # saver = tf.train.import_meta_graph(SAVE_PATH + '/Audiobooks.ckpt' + '.meta')
        # saver.restore(sess, checkpoint)
        # #
        # trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # for var in trainable_var:
        #     print(var.name, var.eval())
        #
        # test_loss = sess.run(loss, feed_dict={'inputs:0': test[0], 'targets:0': test[1]})
        # print(sess.run(pred, feed_dict={'inputs:0': np.random.rand(10, 2)}))
        # print("TEST LOSS = {:0.4f}".format(test_loss))


        # trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # for var in trainable_var:
        #     print(var.name, var.eval())

        # final_weights = tf.get_variable("weights_{0}".format(hidden_layer_size))
        # final_biases = tf.get_variable("biases_{0}".format(hidden_layer_size))

        # train_writer = tf.summary.FileWriter("./models/Audiobooks")
        # train_writer.add_graph(tf.get_default_graph())
        # saver = tf.train.Saver(max_to_keep=1)
        # saver.save(sess, "./models/my_checkpoint.ckpt", global_step=0)

    def predict_with_saved_audiobooks_model(self):

        y_pred = []
        input_size = 10

        tf.reset_default_graph()

        my_inputs = tf.placeholder(tf.float32, [None, input_size])
        test_data = Audiobooks_Data_Reader('test')

        saver = tf.train.import_meta_graph('./models/Audiobooks.meta')

        with tf.Session() as sess:

            # saver.restore(sess, tf.train.latest_checkpoint('./models/'))
            saver.restore(sess, './models/Audiobooks')

            # trainable_var = tf.trainable_variables()
            trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in trainable_var:
                print(var.name)


            for input_batch, target_batch in test_data:
                # print(input_batch)
                output_batch = sess.run([y_pred], feed_dict={my_inputs: input_batch})
                print(output_batch)


# do the batching in a fast and efficient mode
class Audiobooks_Data_Reader():
    def __init__(self, dataset, batch_size=None):
        npz = np.load("./data/Audiobooks_data_{0}.npz".format(dataset))
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size

        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] / self.batch_size

    # a method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1

        # number of classes in the classification problem (here it's just 0s and 1s)
        classes_num = 2
        # one-hot encoded targets
        # 0 = [1,0]
        # 1 = [0,1]
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1

        return inputs_batch, targets_one_hot

    # tells python that the class is an iterator
    def __iter__(self):
        return self


if __name__ == '__main__':
    deeplearn = DeepLearning()
    # deeplearn.test_gradient()
    # deeplearn.test_tensorflow_init()
    # deeplearn.test_tensorflow_solve()

    # deeplearn.read_audiobook_data()
    deeplearn.get_audiobooks_model()
    # deeplearn.predict_with_saved_audiobooks_model()




