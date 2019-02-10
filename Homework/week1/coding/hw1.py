import numpy as np
from sklearn import svm
import sklearn
# %matplotlib inline
import os
import operator


def splitData(_folder, _split=3):
    """
    Looks through items in folder (npy data)
    validation data will be empty if split into 2
    :param _folder: folder to look through
    :param _split: number of data groups to split into
    :return: training data array, validation data array, test data array
    """

    flowers_array = []
    temp_array = []
    training_data = []
    validation_data = []
    final_test_data = []

    # create array of arrays of total data
    count = 0
    for files in os.listdir(_folder):
        temp_array.append(files)

        count += 1
        if count == 80:
            flowers_array.append(temp_array)
            count = 0
            temp_array = []

    # randomise data in each category
    for groups in flowers_array:
        np.random.shuffle(groups)

        if _split == 2:
            training_data.extend(groups[0:60])
            final_test_data.extend(groups[60:80])
        else:
            training_data.extend(groups[0:40])
            validation_data.extend(groups[40:60])
            final_test_data.extend(groups[60:80])

    return training_data, validation_data, final_test_data


def trainModels(_data, _group=0, _c=1.0, _datapoints=40):
    """
    trains a SVM model
    :param _data: array of training data
    :param _group: the group that the data belongs to
    :param _c: c to be used later
    :param _datapoints: number of data in _data set
    :return:
    """

    # Create svm classification object
    model = svm.SVC(kernel='linear', C=_c)

    count = 0
    x_train = []
    y_train = []

    lower_bound = _group * _datapoints
    upper_bound = (_group + 1) * _datapoints

    for file in _data:
        x_train.append(np.load(os.path.join(basepath,file)))
        if lower_bound < count < upper_bound:
            y_train.append(1)
        else:
            y_train.append(0)
        count += 1

    model.fit(x_train, y_train)
    return model


def readData(_data_array):
    """
    Converts data array into usable data
    :param _data_array: array of string for filenames
    :return: train data
    """
    x_data = []

    for file in _data_array:
        x_data.append(np.load(os.path.join(basepath, file)))

    return x_data


def calculateAccuracy(_predicted, _actual):
    acc = sklearn.metrics.accuracy_score(np.array(_predicted),
                                         np.array(_actual))
    return acc


def averageAccuracy(_accuracy_arr):
    total = 0
    for i in _accuracy_arr:
        total += i
    return total/len(_accuracy_arr)


def generateCorrectArray(_group, _size, _datapoints=20):
    actual = []
    for num in range(_size):
        if (_group * _datapoints) < num < (_datapoints * (_group + 1)):
            actual.append(1)
        else:
            actual.append(0)
    return actual


def getBestAccuracy(accuracies, classifiers):
    index, accuracy = max(enumerate(accuracies), key=operator.itemgetter(1))
    return accuracy, classifiers[index]


def compareValues(highest_accuracy, accuracy, best_reg_const, reg_const):
    if highest_accuracy >= accuracy:
        return highest_accuracy, best_reg_const
    return accuracy, reg_const


def getHighestValueForC(training_data, v_data):
    # regularization constants
    REG_CONST = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]

    # variables to determine best
    highest_accuracy = 0
    best_reg_const = 0

    all_accuracies = []
    all_regs = []
    for reg in REG_CONST:

        accuracies = []
        classifiers = []
        for group in range(17):
            # train data
            clf = trainModels(training_data, group, reg)

            # add to array of reg
            classifiers.append(clf)

            # calculate accuracy
            predicted = clf.predict(v_data)
            actual = generateCorrectArray(group, len(predicted), _datapoints=40)
            accuracy = calculateAccuracy(predicted, actual)
            accuracies.append(accuracy)

        # compare accuracies
        accuracy, best_clf = getBestAccuracy(accuracies, classifiers)
        all_accuracies.append(accuracy)
        all_regs.append(reg)
        highest_accuracy, best_reg_const = compareValues(highest_accuracy, accuracy, best_reg_const, reg)

    print("--------- Report ---------")
    print("All accuracies:", all_accuracies)
    print("Corresponding reg_constants:", all_regs)
    print("Highest accuracy:", highest_accuracy)
    print("Best regularizer:", best_reg_const)
    return highest_accuracy, best_reg_const


def classifyTestSet(training_data, test_data):
    REG = 0.01

    accuracies = []
    classifiers = []
    for group in range(17):
        # train data
        clf = trainModels(training_data, group, REG, _datapoints=60)

        classifiers.append(clf)

        # calculate accuracy
        predicted = clf.predict(test_data)
        actual = generateCorrectArray(group, len(predicted))
        accuracy = calculateAccuracy(predicted, actual)
        accuracies.append(accuracy)

    # compare accuracies
    accuracy, best_clf = getBestAccuracy(accuracies, classifiers)
    print("list of accuracies for each class:\n", accuracies)
    print("Highest accuracy:", accuracy)
    return accuracy


if __name__ == '__main__':
    # change this accordingly to path directory of flower features
    basepath = 'flowers17feats/flowers17/feats'

    # split data
    training_data, validation_data, test_data = splitData(basepath)

    # convert data to usable format
    t_data = readData(training_data)
    v_data = readData(validation_data)
    test_data = readData(test_data)

    # highest value for C here is 0.01, with accuracy 0.961764705882353
    getHighestValueForC(training_data, v_data)


    # # re-split data differently
    # training_data, _, test_data = splitData(basepath, 2)
    #
    # test_data = readData(test_data)
    #
    # classifyTestSet(training_data, test_data)
