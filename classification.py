import pandas
import numpy as np
from numpy import array
from Bio.Seq import Seq
from Bio.Cluster import kcluster
from sklearn.model_selection import train_test_split


CLUSTER_NUMBER = 5


def load_data():
    groups = {"g1": [],
              "g2": [],
              "g3": [],
              "g4": [],
              "g5": []}
    x = []
    y = []
    with open("data_by_groups.csv", "r") as data_file:
        for line_index, line in enumerate(data_file.readlines()):
            if line_index != 0:
                for index, value in enumerate(line.strip().split(",")):
                    value = value.strip().replace("*", "")
                    if value != "":
                        groups["g{}".format(index+1)].append(Seq(value))
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


def smart_letter_distance(l1, l2):
    if l1 == l2:
        return 0
    elif l1 == 'A' and l2 == 'T' or l1 == 'T' and l2 == 'A' or l1 == 'C' and l2 == 'G' or l1 == 'G' and l2 == 'C':
        return 1
    else:
        return 2


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strand lengths are not equal!")
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def hamming_distance_smart(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strand lengths are not equal!")
    return sum([smart_letter_distance(ch1, ch2) for ch1, ch2 in zip(s1, s2)])


def run():
    data_groups, x, y = load_data()
    x = fix_length(x)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    clusterid, error, found = make_k_means_unsuperviesed(x)
    clusterid = list(clusterid)
    fit_dict = build_acc_fit_dict(clusterid, y)
    acc = sum([1 if y[index] == fit_dict[clusterid[index]] else 0 for index in range(len(clusterid))]) / len(clusterid)
    print("We train unsupervised kmeans algorithm on {} samples getting {:.2f}% accuracy".format(len(x), 100 * acc))


if __name__ == '__main__':
    run()
