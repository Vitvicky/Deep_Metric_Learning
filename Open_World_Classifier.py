import numpy as np
from scipy import stats
from CNN_Metric.Model_Train import *


class Open_World_Classifier(object):

    def __init__(self, n_class, train_data, train_label):
        self.n_class = n_class
        new_emerge_threshold, self.svm_model = self._compute_threshold(train_data, train_label)
        self.classes_ = {}
        # class_list = [i for i in range(n_class)]
        classes_ = {}
        for i, each in enumerate(sorted(np.unique(train_label))):
            self.classes_[i] = each
            # self.class_idx[each] = i

        self.new_emerge_threshold = np.zeros((len(sorted(self.classes_)),))

        for cls_idx in self.classes_:
            if self.classes_[cls_idx] in new_emerge_threshold:
                self.new_emerge_threshold[cls_idx] = new_emerge_threshold[self.classes_[cls_idx]]
            else:
                self.new_emerge_threshold[cls_idx] = 0

    def _compute_threshold(self, train_data, train_label):
        predict_labels, predict_prob, svm_model = svm_classifier_set(self.n_class, train_data, train_label)

        # probability distribution
        p_dist = {}
        for i in range(len(predict_labels)):
            label = predict_labels[i]
            p = predict_prob[i]

            if label not in p_dist:
                p_dist[label] = []

            p_dist[label].append(p)

        # compute one-side lower confidence bound
        alpha = 0.01

        # compute sample mean and sample variance
        p_stat = {}
        for label in p_dist:
            p_stat[label] = {
                'x_mean': np.mean(p_dist[label]),
                'x_std': np.std(p_dist[label]),
                'n': len(p_dist[label])
            }

        # threshold 1 (lower (1-alpha) confidence bound of prob mean
        threshold1 = {}

        for label in p_stat:
            n = p_stat[label]['n']
            t_value = stats.t.ppf(1 - alpha, n)
            x_mean = p_stat[label]['x_mean']
            s = p_stat[label]['x_std']
            threshold1[label] = x_mean - t_value * s / np.sqrt(n)

        # threshold2
        threshold2 = {}

        for label in p_stat:
            n = p_stat[label]['n']
            chi_value = stats.chi2.ppf(1 - alpha, n)
            mu = p_stat[label]['x_mean']
            s = p_stat[label]['x_std']
            sigma = np.sqrt((n - 1) * (s ** 2) / chi_value)
            threshold2[label] = mu - 1 * sigma

        # determine the threshold
        novel_threshold = threshold1
        return novel_threshold, svm_model

    def instance_predict(self, x, label):
        # one-vs-all classifier
        instance_probability = svm_classifier_instance(x, self.svm_model)
        instance_probability = np.array(instance_probability)

        instance_probability = instance_probability[0]
        # print("emerge_threshold is: ")
        # print(self.new_emerge_threshold)
        # print("real label is: " + str(label))
        # print("predict prob is: ")
        # print(instance_probability)
        # print("---------------------------------------------------")
        # check if novel
        self.new_emerge_threshold[0] = 0.3
        # self.new_emerge_threshold[1] = 0.35
        self.new_emerge_threshold[2] = 0.35
        self.new_emerge_threshold[4] = 0.3
        # self.new_emerge_threshold[5] = 0.85
        assert len(instance_probability) == len(self.new_emerge_threshold)
        # if probability of a particular class is less than the corresponding threshold
        # it is claimed as potential novel class

        # only for fashion-mnist data set
        # condition_check = instance_probability < self.new_emerge_threshold
        #
        # if all(condition_check):
        #     # this is a novel class
        #     p_label = -2
        #     p_prob = 1.0
        # else:
        #     # one-vs-all classifier
        #     # satisfy some classes, find max in these classes
        #     idx = np.arange(len(instance_probability))
        #     remain_idx = idx[np.invert(condition_check)]
        #     remain_prob = instance_probability[np.invert(condition_check)]
        #     max_idx = np.argmax(remain_prob)
        #     class_idx = remain_idx[max_idx]
        #     p_label1 = self.classes_[class_idx]
        #     p_prob1 = instance_probability[class_idx]
        #
        #     p_label = p_label1
        #     p_prob = p_prob1
        idx_max = np.argmax(instance_probability)
        max_prob = instance_probability[idx_max]
        max_threshold = self.new_emerge_threshold[idx_max]
        if max_threshold > max_prob:
            # this is a novel class
            p_label = -2
            p_prob = 1.0
        else:
            p_prob = max_prob
            p_label = self.classes_[idx_max]

        return p_label, p_prob