import os
import logging
from itertools import combinations_with_replacement
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
import CNN_Metric.Network_Metric as ddml


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def train(net, dataloader, criterion, optimizer):
    logger = logging.getLogger(__name__)

    statistics_batch = len(dataloader) / 10

    cnn_loss = 0.0
    ddml_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        # onehot
        # target = torch.zeros(labels.shape[0], 10).scatter_(1, labels.long().view(-1, 1), 1).to(net.device)
        inputs, labels = inputs.to(net.device), labels.to(net.device)
        pairs = list(combinations_with_replacement(zip(inputs, labels), 2))

        ################
        # cnn backward #
        ################
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # bce
        # loss = criterion(outputs, target)
        # cross entropy
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        cnn_loss += loss.item()

        #################
        # ddml backward #
        #################
        ddml_loss += net.ddml_optimize(pairs)

        # print statistics
        if (i + 1) % statistics_batch == 0:
            logger.debug('%5d: nn loss: %.4f, ddml loss: %.4f', i + 1, cnn_loss / statistics_batch, ddml_loss / statistics_batch)
            cnn_loss = 0.0
            ddml_loss = 0.0


def svm_test(net, trainloader, testloader):
    svm_test.svc = svm.SVC(kernel='linear', C=10, gamma=0.1)

    train_x = []
    train_y = []
    for x, y in trainloader:
        x, y = x.to(net.device), y.to(net.device)
        x = net.ddml_forward(x)
        x = x.to(torch.device('cpu'))
        train_x.append(x.squeeze().detach().numpy())
        train_y.append(int(y))

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    svm_test.svc.fit(train_x, train_y)

    test_x = []
    test_y = []
    for x, y in testloader:
        x, y = x.to(net.device), y.to(net.device)
        x = net.ddml_forward(x)
        x = x.to(torch.device('cpu'))
        test_x.append(x.squeeze().detach().numpy())
        test_y.append(int(y))

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    predictions = svm_test.svc.predict(test_x)

    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

    return accuracy, cm


def svm_classifier_set(n_class, train_features, train_labels):
    classes_ = {}
    # dataset comes from TestDataset(size=n)
    # assign dict class_
    for i, each in enumerate(sorted(np.unique(train_labels))):
            classes_[i] = each

    train_labels = np.array(train_labels)
    train_features = np.array(train_features)
    svc = svm.SVC(kernel='linear', C=32, probability=True)
    svc.fit(train_features, train_labels)

    predict_probs = svc.predict_proba(train_features)
    predict_probs = np.array(predict_probs)

    # find max probability
    max_prob = np.max(predict_probs, axis=1)
    max_idx = np.argmax(predict_probs, axis=1)
    predicted_labels = []
    predicted_prob = []
    for i in range(predict_probs.shape[0]):
        predicted_labels.append(classes_[max_idx[i]])
        predicted_prob.append(max_prob[i])
        # print('{}： "{}"'.format('Predict class is: ', classes_[max_idx[i]]))
        # print('{}： "{}"'.format('Max Probability is: ', max_prob[i]))
        # print('{}： "{}"'.format('Real label is: ', data_set.labels[i]))
        # print("---------------------------------------------------------------")
    return predicted_labels, predicted_prob, svc


def svm_classifier_instance(instance, svm_model):
    instances = []
    instances.append(instance)
    probability = svm_model.predict_proba(np.array(instances))
    return probability

if __name__ == '__main__':
    LOGGER = setup_logger(level=logging.DEBUG)

    TRAIN_BATCH_SIZE = 5
    TRAIN_EPOCH_NUMBER = 30
    TRAIN_TEST_SPLIT_INDEX = 6000

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")

    ###############
    # csv dataset #
    ###############
    DATASET = np.loadtxt(ddml.DATASET_PATH, delimiter=',')
    np.random.shuffle(DATASET)
    LOGGER.debug("Dataset shape: %s", DATASET.shape)

    TRAINSET = ddml.DDMLDataset(DATASET[:TRAIN_TEST_SPLIT_INDEX])
    TRAINLOADER = DataLoader(dataset=TRAINSET, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    SVM_TRAINLOADER = DataLoader(dataset=TRAINSET, batch_size=1, shuffle=True, num_workers=4)

    TESTSET = ddml.DDMLDataset(DATASET[TRAIN_TEST_SPLIT_INDEX:])
    TESTLOADER = DataLoader(dataset=TESTSET, batch_size=1, shuffle=False, num_workers=4)

    cnnnet = ddml.DDMLNet(device=DEVICE, beta=0.5, tao=10.0, b=2.0, learning_rate=0.0003)

    cross_entropy = nn.CrossEntropyLoss()
    # bce = nn.BCELoss()
    sgd = optim.SGD(cnnnet.parameters(), lr=0.0003, momentum=0.9)

    if os.path.exists(ddml.PKL_PATH):
        state_dict = torch.load(ddml.PKL_PATH)
        try:
            cnnnet.load_state_dict(state_dict)
            LOGGER.info("Load state from file %s.", ddml.PKL_PATH)
        except RuntimeError:
            LOGGER.error("Loading state from file %s failed.", ddml.PKL_PATH)

    for epoch in range(TRAIN_EPOCH_NUMBER):
        LOGGER.info("Trainset size: %d, Epoch number: %d", len(TRAINSET), epoch + 1)
        train(cnnnet, TRAINLOADER, criterion=cross_entropy, optimizer=sgd)

        if (epoch + 1) % 5 == 0:
            LOGGER.info("Testset size: %d", len(TESTSET))
            svm_accuracy, svm_cm = svm_test(cnnnet, SVM_TRAINLOADER, TESTLOADER)
            LOGGER.info("SVM Accuracy: %6f", svm_accuracy)
            torch.save(cnnnet.state_dict(), '/home/wzy/PycharmProjects/DDML/pkl/ddml-({:.4f}).pkl'.format(svm_accuracy))
            torch.save(cnnnet.state_dict(), '/home/wzy/PycharmProjects/DDML/pkl/ddml-guardian.pkl')
