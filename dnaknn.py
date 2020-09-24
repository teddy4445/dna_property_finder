class DnaKNN:

    def __init__(self,
                 x_train,
                 y_train):
        self.groups = {}
        self.x_train = x_train
        self.y_train = y_train
        for index in range(len(x_train)):
            try:
                self.groups[y_train[index]].append(x_train[index])
            except:
                self.groups[y_train[index]] = [x_train[index]]

    def calc_acc(self, x_test, y_test, metric="smart"):
        y_pred = [self.classify(x=item, metric=metric) for item in x_test]
        return sum([y_test[index] == y_pred[index] for index in range(len(y_test))]) / len(y_test)

    def k_calc_acc(self, x_test, y_test, metric="smart", k=5):
        y_pred = [self.k_classify(x=item, metric=metric, k=k) for item in x_test]
        return sum([y_test[index] == y_pred[index] for index in range(len(y_test))]) / len(y_test)

    def classify(self, x, metric="smart"):
        best_dist = 9999999
        best_class = 0
        for group in self.groups:
            new_dist = self.cluster_dist(self.groups[group],
                                         x,
                                         metric=metric)
            if new_dist < best_dist:
                best_dist = new_dist
                best_class = group
        return best_class

    def k_classify(self, x, metric="smart", k=5):
        dist_class = []
        for x_train_item_index in range(len(self.x_train)):
            new_dist = 0
            if metric == "hamming":
                new_dist = DnaKNN.hamming_distance(self.x_train[x_train_item_index], x)
            elif metric == "smart":
                new_dist = DnaKNN.hamming_distance_smart(self.x_train[x_train_item_index], x)
            dist_class.append([new_dist, self.y_train[x_train_item_index]])
        dist_class = sorted(dist_class, key=lambda x: x[0])
        groups = [dist_class[i][1] for i in range(k)]
        return DnaKNN.most_frequent(groups)

    def cluster_dist(self, cluster, x, metric="smart"):
        dist = 0
        for member in cluster:
            new_dist = 0
            if metric == "hamming":
                new_dist = DnaKNN.hamming_distance(member, x)
            elif metric == "smart":
                new_dist = DnaKNN.hamming_distance_smart(member, x)
            dist += new_dist
        dist /= len(cluster)
        return dist

    @staticmethod
    def smart_letter_distance(l1, l2):
        if l1 == l2:
            return 0
        elif l1 == 'A' and l2 == 'T' or l1 == 'T' and l2 == 'A' or l1 == 'C' and l2 == 'G' or l1 == 'G' and l2 == 'C':
            return 1
        else:
            return 2

    @staticmethod
    def hamming_distance(s1, s2):
        if len(s1) != len(s2):
            raise ValueError("Strand lengths are not equal!")
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

    @staticmethod
    def hamming_distance_smart(s1, s2):
        if len(s1) != len(s2):
            raise ValueError("Strand lengths are not equal!")
        return sum([DnaKNN.smart_letter_distance(ch1, ch2) for ch1, ch2 in zip(s1, s2)])

    @staticmethod
    def most_frequent(val_list):
        dict = {}
        count, itm = 0, ''
        for item in reversed(val_list):
            dict[item] = dict.get(item, 0) + 1
            if dict[item] >= count:
                count, itm = dict[item], item
        return itm
