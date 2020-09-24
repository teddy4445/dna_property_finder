import pandas
import numpy as np
from numpy import array
from Bio.Seq import Seq
from Bio.Cluster import kcluster
from sklearn.model_selection import train_test_split

# project imports
from dnaknn import DnaKNN

CLUSTER_NUMBER = 5
MIN_DNA_LENGTH = 220
IS_MAJOR_CLASSES_ONLY = True


def load_data():
    groups = {}
    x = []
    y = []
    with open("data_by_groups.csv", "r") as data_file:
        for line_index, line in enumerate(data_file.readlines()):
            if line_index != 0:
                for index, value in enumerate(line.strip().split(",")):
                    value = value.strip().replace("*", "")
                    if value != "" and ((IS_MAJOR_CLASSES_ONLY and index in [0, 2]) or not IS_MAJOR_CLASSES_ONLY):
                        try:
                            groups["g{}".format(index + 1)].append(Seq(value[:MIN_DNA_LENGTH]))
                        except:
                            groups["g{}".format(index + 1)] = [Seq(value[:MIN_DNA_LENGTH])]
                        x.append(Seq(value))
                        y.append(index+1)
    return groups, x, y


def fix_length(x):
    min_length = min(map(len, x))
    return [item[:min_length] for item in x]


def build_acc_fit_dict(clusterid, y):
    answer = {index: index + 1 for index in range(CLUSTER_NUMBER)}
    hit_flags = [False for i in range(CLUSTER_NUMBER)]
    run_index = 0
    while not all(hit_flags):
        hit_flags[clusterid[run_index]] = True
        answer[clusterid[run_index]] = y[run_index]
        run_index += 1
        if run_index == len(clusterid):
            break
    return answer


def make_k_means_unsuperviesed(x):
    matrix = np.array([np.fromstring(seq._data, dtype=np.uint8) for seq in x])
    clusterid, error, found = kcluster(matrix,
                                       nclusters=CLUSTER_NUMBER)
    return clusterid, error, found


def train_test_split_by_groups(data_groups,
                               test_size=0.3):
    final_x_train = []
    final_y_train = []
    final_x_test = []
    final_y_test = []
    for group_key in data_groups:
        x = data_groups[group_key]
        y = [int(group_key[-1]) for i in range(len(x))]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        final_x_train.extend(x_train)
        final_y_train.extend(y_train)
        final_x_test.extend(x_test)
        final_y_test.extend(y_test)
    return final_x_train, final_y_train, final_x_test, final_y_test


def run_unsupervised():
    data_groups, x, y = load_data()
    x = fix_length(x)
    clusterid, error, found = make_k_means_unsuperviesed(x)
    clusterid = list(clusterid)
    fit_dict = build_acc_fit_dict(clusterid, y)
    acc = sum([1 if y[index] == fit_dict[clusterid[index]] else 0 for index in range(len(clusterid))]) / len(clusterid)
    print("We train unsupervised kmeans algorithm on {} samples getting {:.2f}% accuracy".format(len(x), 100 * acc))


def run_supervised():
    run_supervised_k_fold(k=1)


def run_supervised_k_fold(k: int = 10, knn_k: int = 5):
    data_groups, x, y = load_data()
    final_acc_smart = []
    final_acc_hamming = []
    final_k_acc_smart = []
    final_k_acc_hamming = []
    for repeat_index in range(k):
        x_train, y_train, x_test, y_test = train_test_split_by_groups(data_groups=data_groups,
                                                                      test_size=0.32)
        model = DnaKNN(x_train, y_train)
        if False:
            acc_smart = model.calc_acc(x_test=x_test,
                                       y_test=y_test,
                                       metric="smart")
            acc_hamming = model.calc_acc(x_test=x_test,
                                         y_test=y_test,
                                         metric="hamming")
        k_acc_smart = model.k_calc_acc(x_test=x_test,
                                     y_test=y_test,
                                     metric="smart",
                                     k=knn_k)
        k_acc_hamming = model.k_calc_acc(x_test=x_test,
                                       y_test=y_test,
                                       metric="hamming",
                                       k=knn_k)
        if False:
            final_acc_smart.append(acc_smart)
            final_acc_hamming.append(acc_hamming)
        final_k_acc_smart.append(k_acc_smart)
        final_k_acc_hamming.append(k_acc_hamming)
        print("Finish test (#{})".format(repeat_index+1))

    if False:
        acc_smart = sum(final_acc_smart) / len(final_acc_smart)
        acc_hamming = sum(final_acc_hamming) / len(final_acc_hamming)
    k_acc_smart = sum(final_k_acc_smart) / len(final_k_acc_smart)
    k_acc_hamming = sum(final_k_acc_hamming) / len(final_k_acc_hamming)

    if k > 1:
        print("\n\nResults for k-fold test (k={})".format(k))

    if False:
        print("We train supervised KNN algorithm [smart metric] on {} samples, tested on {} and getting {:.2f}% accuracy".format(len(x_train), len(x_test), 100 * acc_smart))
        print("We train supervised KNN algorithm [hamming metric] on {} samples, tested on {} and getting {:.2f}% accuracy".format(len(x_train), len(x_test), 100 * acc_hamming))
    print("[k = {}] We train supervised KNN algorithm [smart metric] on {} samples, tested on {} and getting {:.2f}% accuracy".format(knn_k, len(x_train), len(x_test), 100 * k_acc_smart))
    print("[k = {}] We train supervised KNN algorithm [hamming metric] on {} samples, tested on {} and getting {:.2f}% accuracy".format(knn_k, len(x_train), len(x_test), 100 * k_acc_hamming))
    return k_acc_hamming, k_acc_smart


if __name__ == '__main__':
    #run_unsupervised()
    answer = []
    for k in range(3, 10):
        k_acc_hamming, k_acc_smart = run_supervised_k_fold(k=5, knn_k=k)
        answer.append([k, k_acc_smart, k_acc_hamming])
    print("\n\n\n\n\n")
    for i in range(len(answer)):
        print("For KNN's k = {}, smart metric = {:.2f} and hamming metric = {:.2f}\n".format(answer[i][0], answer[i][1], answer[i][2]), end="")
