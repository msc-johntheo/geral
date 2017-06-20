import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.debugger import Pdb
from sklearn.neural_network import MLPClassifier
import tensorflow as tf


def extract_feature(file_name):
    '''
    Extrai as features que serão utilizadas como entrada na rede neural
    '''
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(name, parent_dir, sub_dirs, file_ext="*.wav"):
    '''
    Obtem o arquivo de audio do diretorio, extrai as features e grava em um array
    '''
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print("Error encountered while parsing file: {}".format(fn))
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('\\')[2].split('-')[1])
    features_array = np.array(features)
    labels_array = np.array(labels, dtype=np.int)
    result = np.array([np.append(features_array[i], labels_array[i]) for i in range(len(features_array))])
    np.savetxt("{}.csv".format(name), result, delimiter=",")
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    '''
    Tranforma os labels em um elemento one-hot, ou seja, uma sequencia de zeros(0) onde apenas um elemento é hum(1)
    :param labels: 
    :return: 
    '''
    n_labels = len(labels)
    # n_unique_labels = len(np.unique(labels))
    # one_hot_encode = np.zeros((n_labels, n_unique_labels))
    max_value_label = 10
    one_hot_encode = np.zeros((n_labels, max_value_label))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def generate_data():
    parent_dir = 'audio'
    tr_sub_dirs = ["fold4"]
    ts_sub_dirs = ["fold6"]
    tr_features, tr_labels = parse_audio_files("train-fold4", parent_dir, tr_sub_dirs)
    # ts_features, ts_labels = parse_audio_files("test-fold6", parent_dir, ts_sub_dirs)

    # return tr_features, tr_labels, ts_features, ts_labels


def run():
    train_data = np.genfromtxt('full-train.csv', delimiter=',')
    test_data = np.genfromtxt('test-fold6.csv', delimiter=',')

    train_test_split = int(len(train_data)*0.70)

    X_train, y_train = train_data[:train_test_split, :193], train_data[:train_test_split, 193]
    # X_test, y_test = test_data[:, :193], test_data[:, 193]
    X_test, y_test = train_data[train_test_split:, :193], train_data[train_test_split:, 193]

    mlp = MLPClassifier(hidden_layer_sizes=(300, 300,),
                        activation='relu',
                        solver='sgd',
                        # beta_1=0.9,
                        # beta_2=0.999,
                        # epsilon=1e-08,
                        verbose=False,
                        tol=0.001,
                        shuffle=True,
                        random_state=13,
                        early_stopping=True,
                        validation_fraction=0.2,
                        learning_rate='constant',
                        learning_rate_init=0.0001)

    # mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
    ##              beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #             hidden_layer_sizes=(300,300), learning_rate='constant',
    #             learning_rate_init=1, max_iter=200, momentum=0.9,
    #              nesterovs_momentum=True, power_t=0.5, random_state=None,
    #              shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
    #              verbose=False, warm_start=False)

    mlp.fit(X_train, y_train)
    score_treino = round(mlp.score(X_train, y_train), 2)
    score_test = round(mlp.score(X_test, y_test), 2)

    print("Score treino: {} | Score test: {} | Epochs: {} | Classes: {} | Outputs: {}"
          .format(score_treino, score_test, mlp.n_iter_, mlp.classes_, mlp.n_outputs_))

    plt.plot(mlp.loss_curve_, label='Loss Curve')
    plt.show()


def run_tf(tr_features):
    training_epochs = 50
    n_dim = tr_features.shape[1]
    n_classes = 10
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.01


def main():
    run()


if __name__ == "__main__":
    main()
