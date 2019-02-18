from __future__ import print_function, division

import os
import shutil
import sys
import time
import numpy as np
import tensorflow as tf
import util as util
from tensorflow.python.saved_model import tag_constants


class Model:
    def __init__(self,
                 hidden_size=128,
                 batch_size=5,
                 num_feat=4,
                 num_class=3,
                 num_epochs=20,
                 trunc_len=25,
                 layers=3,
                 lr=0.005,
                 dropout_prob=0.1):

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_feat = num_feat
        self.num_class = num_class
        self.num_epochs = num_epochs
        self.trunc_len = trunc_len
        self.layers = layers
        self.lr = lr
        self.dropout_prob = dropout_prob

        self.batchX_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.trunc_len, self.num_feat])
        self.mask_placeholder = tf.placeholder(tf.bool, [self.batch_size, self.trunc_len])
        self.batchY_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.trunc_len])
        self.predX_placeholder = tf.placeholder(tf.float32, [1, None, self.num_feat], name="prediction_x")
        self.predY_placeholder = tf.placeholder(tf.int32, [1, None], name="prediction_y")
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        self.w = tf.Variable(xavier_initializer([self.hidden_size, self.num_class]), dtype=tf.float32, name="w")
        self.b = tf.Variable(xavier_initializer([self.num_class]), dtype=tf.float32, name="b")

        if self.layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(
                    self.hidden_size) for _ in range(self.layers)])
        else:
            self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)  # , kernel_initializer=init, bias_initializer=init)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.dropout_prob)

        self.c_placeholder = [tf.placeholder(tf.float32, [self.batch_size, self.hidden_size])
                              for _ in range(self.layers)]
        self.h_placeholder = [tf.placeholder(tf.float32, [self.batch_size, self.hidden_size])
                              for _ in range(self.layers)]

        self.pred = self.forward_pass()
        self.corr_pred = self.correct_predictions(self.pred[0])
        self.loss = self.losses(self.pred[0])
        self.train = self.train_op(self.loss)
        self.single_prediction = self._single_prediction()

    def forward_pass(self):
        """
        Output: logits of the input
                shape = (batch_size, input_length, num_classes)
        """

        state = [tf.nn.rnn_cell.LSTMStateTuple(self.c_placeholder[idx], self.h_placeholder[idx]) for idx in range(self.layers)]
        outputs, state \
            = tf.nn.dynamic_rnn(self.cell, self.batchX_placeholder, dtype=tf.float32,
                                initial_state=tuple(state), swap_memory=True)
        logits = tf.einsum("bnm,mp->bnp", outputs, self.w) + self.b
        return logits, state

    def correct_predictions(self, logits):
        predictions = tf.nn.softmax(logits)
        correct_predictions = tf.equal(tf.cast(tf.argmax(predictions, 2), tf.int32), self.batchY_placeholder)
        return predictions, correct_predictions

    def losses(self, logits):
        total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.batchY_placeholder)
        actual_loss = tf.reduce_mean(tf.boolean_mask(total_loss, self.mask_placeholder))
        return actual_loss

    def train_op(self, loss):
        return tf.train.AdagradOptimizer(self.lr).minimize(loss)

    def _single_prediction(self):
        init_state = self.cell.zero_state(1, tf.float32)
        outputs, state \
                = tf.nn.dynamic_rnn(self.cell, self.predX_placeholder, dtype=tf.float32, swap_memory=True,
                                    initial_state=init_state)
        logits = tf.add(tf.einsum("bnm,mp->bnp", outputs, self.w), self.b, name="logits")
        predictions = tf.argmax(tf.nn.softmax(logits), 2, name="predictions")
        correct_predictions = tf.equal(tf.cast(predictions, tf.int32), self.predY_placeholder, name="correct_predictions")
        return logits, predictions, correct_predictions

    def create_feed_dict(self, state, x_, y_, mask_):
        dict1 = {self.h_placeholder[idx]: state[idx][0] for idx in range(self.layers)}
        dict2 = {self.c_placeholder[idx]: state[idx][1] for idx in range(self.layers)}
        return {**dict1, **dict2,
                self.batchX_placeholder: x_,
                self.batchY_placeholder: y_,
                self.mask_placeholder: mask_}

    def run(self, sess, x_train, y_train, mask):
        data_len = len(x_train[1])
        loss_per_epoch = []
        acc_per_epoch = []

        # Start params for early stopping
        current_best_loss = np.inf
        patience = 0

        for epoch in range(self.num_epochs):
            starttime = time.time()
            state = self.cell.zero_state(self.batch_size, tf.float32)
            state = sess.run(state)
            print("Epoch ", epoch)
            correct_pred = None
            loss = 0

            # Truncated training iterations
            total_truncations = int(data_len/self.trunc_len)
            for idx in range(total_truncations):
                util.progress_bar(idx+1, total_truncations)
                start_idx = idx*self.trunc_len
                end_idx = (idx+1)*self.trunc_len
                x_batch = x_train[:, start_idx:end_idx, :]
                y_batch = y_train[:, start_idx:end_idx]
                mask_batch = mask[:, start_idx:end_idx]

                (_logits, state), (predictions, _correct_pred), _total_loss, _train_step = sess.run(
                    [self.pred, self.corr_pred, self.loss, self.train],
                    feed_dict=self.create_feed_dict(state, x_batch, y_batch, mask_batch)
                )

                # Accumulate results for each truncated batch
                if correct_pred is None:
                    correct_pred = _correct_pred
                else:
                    correct_pred = np.concatenate((correct_pred, _correct_pred), 0)
                loss += _total_loss

            # Calculate and print epoch results
            reshaped = np.reshape(correct_pred, [-1])
            accuracy = np.mean(reshaped)
            print("\n\tloss: ", loss)
            print("\taccuracy: ", accuracy)
            print("\ttime elapsed: ", time.time() - starttime)
            loss_per_epoch.append(loss)
            acc_per_epoch.append(accuracy)

            # Save model at each epoch
            (logits, predictions, correct_predictions) = self.single_prediction
            print("\nSaving model")
            path = os.path.join(os.getcwd(), "saves/simple_save_epoch_{}".format(epoch))
            shutil.rmtree(path, ignore_errors=True)
            inputs_dict = {
                "prediction_x": self.predX_placeholder,
                "prediction_y": self.predY_placeholder
            }
            outputs_dict = {
                "logits": logits,
                "predictions": predictions,
                "correct_predictions": correct_predictions
            }
            tf.saved_model.simple_save(
                sess, path, inputs_dict, outputs_dict
            )
            print("Ok")

            # Early stopping
            if loss < current_best_loss:
                current_best_loss = loss
                patience = 0
            else:
                if patience > 6:
                    print("\nEarly stopping criteria met, stopping training...\n")
                    break
                patience += 1

        return loss_per_epoch, acc_per_epoch

    def predict(self, sess, x_file, y_file, dest=None):
        print("Predicting on {}, with true annotation {}...".format(x_file, y_file))
        (logits, pred, corr_pred) = sess.run(
            self.single_prediction,
            feed_dict={
                self.predX_placeholder: np.reshape(np.array(util.read_fasta_file(x_file)), (1, -1, self.num_feat)),
                self.predY_placeholder: np.reshape(np.array(util.read_fasta_file(y_file)), (1, -1))
            })
        reshaped = np.reshape(corr_pred, [-1])
        accuracy = np.mean(reshaped)
        print(x_file, "\n\tAccuracy: ", accuracy)
        if dest is None:
            dest = x_file + "_prediction"
        util.write_prediction_to_file(pred[0], dest)


def restore_model(x_path, y_path):
    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2, config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
            path = os.path.join(os.getcwd(), "saves/simple_save_epoch_79")
            print("\nRestoring...")
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                path
            )
            print("Ok")

            print("Predicting on ", x_path, " with annotation ", y_path)
            # Get restored placeholders
            prediction_x = graph2.get_tensor_by_name("prediction_x:0")
            prediction_y = graph2.get_tensor_by_name("prediction_y:0")
            # Get restored model output
            restored_pred = graph2.get_tensor_by_name("logits:0")

            logits = sess.run(restored_pred, feed_dict={
                prediction_x: np.reshape(np.array(util.read_fasta_file(x_path)), (1, -1, 5)),
                prediction_y: np.reshape(np.array(util.read_fasta_file(y_path)), (1, -1))
            })
            predictions = tf.argmax(tf.nn.softmax(logits), 2).eval()

            util.write_prediction_to_file(predictions[0], "data/genome6.fa_restored")
            print("Success!")


def train_model():
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
            # Set model parameters
            batch_size = 7
            model = Model(batch_size=batch_size, num_feat=5, num_class=4, trunc_len=20, lr=0.005, dropout_prob=0.1,
                          num_epochs=80)
            initializer = tf.global_variables_initializer()
            sess.run(initializer)

            # Prepare data by right-padding with 0"s and create boolean mask for the padded inputs
            x_train_ = [util.read_fasta_file("data/genome{0}.fa".format(i)) for i in range(1, batch_size + 1)]
            y_train_ = [util.read_fasta_file("data/true-ann{0}.fa".format(i)) for i in range(1, batch_size + 1)]
            max_len = 0
            for y in y_train_:
                l = len(y)
                if l > max_len:
                    max_len = l
            max_len += 1
            x_train = np.array([np.concatenate((x, [[1,0,0,0,0]]*(max_len - len(x)))) for x in x_train_])
            y_train = np.array([np.append(y, [0]*(max_len - len(y))) for y in y_train_])
            mask = np.array([np.append([1]*len(x), [0]*(max_len-len(x))) for x in x_train_])

            # Train model
            (losses, accuracies) = model.run(sess, x_train, y_train, mask)
            # Write training results to file
            with open("out/training_results.txt", "w") as f:
                f.write("Loss                          | Accuracy\n")
                for (loss, acc) in zip(losses, accuracies):
                    loss_to_write = str(round(loss, 7))
                    loss_len = len(loss_to_write)
                    gap = " " * (30 - loss_len)
                    f.write("{}{}| {}\n".format(loss_to_write, gap, str(round(acc, 7))))

            # Predict on all the genes, including the ones trained on
            for i in range(1, 11):
                model.predict(sess, "data/genome{}.fa".format(i), "data/true-ann{}.fa".format(i))

            # Save the final model
            (logits, predictions, correct_predictions) = model.single_prediction
            print("\nSaving model")
            path = os.path.join(os.getcwd(), "final_model")
            shutil.rmtree(path, ignore_errors=True)
            inputs_dict = {
                "prediction_x": model.predX_placeholder,
                "prediction_y": model.predY_placeholder
            }
            outputs_dict = {
                "logits": logits,
                "predictions": predictions,
                "correct_predictions": correct_predictions
            }
            tf.saved_model.simple_save(
                sess, path, inputs_dict, outputs_dict
            )
            print("Ok")



if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    argv = sys.argv[1:]
    if len(argv) == 3:
        y = argv.pop()
        x = argv.pop()
        arg = argv.pop()
        if arg == "-restore":
            restore_model(x, y)
    elif len(argv) == 0:
        train_model()
    else:
        print("command line argument must be empty or -restore or -hyperpar")
        sys.exit(2)
