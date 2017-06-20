import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('ggplot')


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def get_data():
    train_data = np.genfromtxt('full-train.csv', delimiter=',')
    test_data = np.genfromtxt('test-fold6.csv', delimiter=',')

    train_x, train_y = train_data[:, :193], np.array(train_data[:, 193], dtype=np.int)
    test_x, test_y = test_data[:, :193], np.array(test_data[:, 193], dtype=np.int)

    train_y = one_hot_encode(train_y)
    test_y = one_hot_encode(test_y)

    return train_x, train_y, test_x, test_y


def run_nn(train_x, train_y, test_x, test_y, training_epochs, learning_rate):
    n_dim = train_x.shape[1]
    n_classes = 10
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    training_epochs = training_epochs
    learning_rate = learning_rate

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
    h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

    cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    cost_history = np.empty(shape=[1], dtype=float)
    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            _, cost = sess.run([optimizer, cost_function], feed_dict={X: train_x, Y: train_y})
            cost_history = np.append(cost_history, cost)

        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(test_y, 1))
    return y_pred, y_true, cost_history


def main():
    training_epochs = 10000
    learning_rate = 0.01
    train_x, train_y, test_x, test_y = get_data()
    y_pred, y_true, cost_history = run_nn(train_x, train_y, test_x, test_y, training_epochs, learning_rate)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')

    print("F-Score: {}".format(round(f, 3)))

    plot_cost(cost_history, training_epochs)


def plot_cost(cost_history, training_epochs):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(cost_history)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    plt.show()


if __name__ == "__main__":
    main()
