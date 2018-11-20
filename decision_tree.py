import math
import os
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

DATA_ROOT = '/Users/xana/Dev/data/'
HOT_DATA = 'ivy_l4_hot_entry'
NORMAL_DATA = 'ivy_l4_not_hot_entry'


def load_data_set(data_name, target):
    with open(os.path.join(DATA_ROOT, data_name)) as f:
        for i, line in enumerate(f):
            pass
    features = np.zeros((i + 1, 10))
    with open(os.path.join(DATA_ROOT, data_name)) as f:
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(' ')
            tokens[4] = int(tokens[4] == 'True')
            tokens[6] = int(tokens[6] == 'M')
            tokens[9] = int(tokens[9] == 'True')
            features[i] = tokens
    return features, [target] * features.shape[0]


def normalize_data_set(data):
    for i in range(0, 10):
        data[:, i] = preprocessing.minmax_scale(data[:, i])
    return data


def entropy(data):
    total = len(data)
    count1 = sum(t > 0.5 for t in data)
    count2 = total - count1
    return -count1 / total * math.log2(count1 / total) - count2 / total * math.log2(count2 / total)


def score_classifier(data, target, classifier, name, split=0.25):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=split)
    classifier.fit(X_train, y_train)
    # y_predict = classifier.predict(X_test)
    # print("%s分类" % name)
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predict))
    # print(classification_report(y_test, y_predict, target_names=['Hot', 'Normal']))
    if hasattr(classifier, 'staged_predict'):
        err = np.zeros((150,))
        for i, y_pred in enumerate(classifier.staged_predict(X_test)):
            err[i] = metrics.zero_one_loss(y_pred, y_test)
        return err
    elif hasattr(classifier, 'score'):
        return 1.0 - classifier.score(X_test, y_test)
    # return metrics.zero_one_loss(y_test, y_predict)


if __name__ == '__main__':
    hot_data, hot_target = load_data_set(HOT_DATA, 1)
    normal_data, normal_target = load_data_set(NORMAL_DATA, 0)
    data = normalize_data_set(np.concatenate((hot_data, normal_data)))
    target = hot_target + normal_target
    print('data preprocessing completed.')
    print('原集合信息熵: ', entropy(target))

    scores = np.zeros((3, len(range(5, 50))))
    dtc = tree.DecisionTreeClassifier()
    rtc = RandomForestClassifier()
    abc = AdaBoostClassifier()
    dtc_err = score_classifier(data, target, dtc, "决策树")
    rtc_err = score_classifier(data, target, rtc, "随机森林")
    abc_err = score_classifier(data, target, abc, "Adaboost DT")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, len(abc_err)], [dtc_err] * 2, label='DecisionTree Test Error', color='red')
    ax.plot([0, len(abc_err)], [rtc_err] * 2, label='RandomForest Test Error', color='blue')
    ax.plot(range(0, len(abc_err)), abc_err, label='Adaboost DT Test Error', color='green')
    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.show()
