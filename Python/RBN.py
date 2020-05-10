import pandas as pd
import numpy as np
import random


class RBN(object):
    """
        Radial Basis Network
        - Weighted sum to hidden layer
        - pasted through a normalizing function (normal dis)
        - Weighted sum to output
    """
    predictors = []
    response = []
    c = []
    node = []
    w = []
    adjustment = .1556
    test_train_ratio = .8

    def predict(self, node, w):
        return sum([node[i] * w[i] for i in range(4)])

    def solve(self, data, alpha=.5):
        """
            Solving thru Gradient Decent - post nodes

        :param data: only open, high, low, close fields are used
        :param alpha: learning rate
        :return:
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]

        # 4 nodes x 3 = 12 weights --- MLR type out 4 to one
        self.node = [0 for i in range(4)]
        self.w = [.01 for i in range(16)]

        # Node average and std
        self.c = [[] for i in range(4)]
        for i in range(len(self.predictors)):
            for j in [0, 1, 2]:
                self.c[0].append(self.predictors[i][j])
            for j in [0, 1, 3]:
                self.c[1].append(self.predictors[i][j])
            for j in [0, 2, 3]:
                self.c[2].append(self.predictors[i][j])
            for j in [1, 2, 3]:
                self.c[3].append(self.predictors[i][j])

        for epoch in range(10):
            # Forward Pass
            for i in range(len(self.predictors)):
                # (1) - 1, 2, 3
                self.node[0] = self.predictors[i][0] * self.w[0] + self.predictors[i][1] * self.w[1] \
                    + self.predictors[i][2] * self.w[2]
                self.node[0] = np.power(np.e, -np.power(abs(self.node[0] - np.average(self.c[0]) / np.std(self.c[0])), 2))
                # (2) - 1, 2, 4
                self.node[1] = self.predictors[i][0] * self.w[3] + self.predictors[i][1] * self.w[4] \
                    + self.predictors[i][3] * self.w[5]
                self.node[1] = np.power(np.e, -np.power(abs(self.node[1] - np.average(self.c[1]) / np.std(self.c[1])), 2))
                # (3) - 1, 3, 4
                self.node[2] = self.predictors[i][0] * self.w[6] + self.predictors[i][2] * self.w[7] \
                    + self.predictors[i][3] * self.w[8]
                self.node[2] = np.power(np.e, -np.power(abs(self.node[2] - np.average(self.c[2]) / np.std(self.c[2])), 2))
                # (4) - 2, 3, 4
                self.node[3] = self.predictors[i][1] * self.w[9] + self.predictors[i][2] * self.w[10] \
                    + self.predictors[i][3] * self.w[11]
                self.node[3] = np.power(np.e, -np.power(abs(self.node[3] - np.average(self.c[3]) / np.std(self.c[3])), 2))

                # Final
                output = 0
                for j in range(4):
                    output += self.node[j] * self.w[j+12]

                # Backprop - pre output
                for j in range(12, 16):
                    self.w[j] -= alpha * (output - self.response[i]) * self.node[j - 12]

    def predict(self, input):
        node = [0 for i in range(4)]

        # (1) - 1, 2, 3
        node[0] = input[0] * self.w[0] + input[1] * self.w[1] + input[2] * self.w[2]
        node[0] = np.power(np.e, -np.power(abs(node[0] - np.average(self.c[0]) / np.std(self.c[0])), 2))
        # (2) - 1, 2, 4
        node[1] = input[0] * self.w[3] + input[1] * self.w[4] + input[3] * self.w[5]
        node[1] = np.power(np.e, -np.power(abs(node[1] - np.average(self.c[1]) / np.std(self.c[1])), 2))
        # (3) - 1, 3, 4
        node[2] = input[0] * self.w[6] + input[2] * self.w[7] + input[3] * self.w[8]
        node[2] = np.power(np.e, -np.power(abs(node[2] - np.average(self.c[2]) / np.std(self.c[2])), 2))
        # (4) - 2, 3, 4
        node[3] = input[1] * self.w[9] + input[2] * self.w[10] + input[3] * self.w[11]
        node[3] = np.power(np.e, -np.power(abs(node[3] - np.average(self.c[3]) / np.std(self.c[3])), 2))

        # Final
        output = 0
        for j in range(4):
            output += node[j] * self.w[j + 12]

        return output

    def validate(self):
        """
        :return: printed summary of model on full data set
        """
        errors = []
        for i in range(round(len(self.predictors) * self.test_train_ratio), len(self.predictors)):
            errors.append(self.response[i] - self.predict(self.predictors[i]) - self.adjustment)
        print("\n=================== Results ====================\n"
              "Mean Summed Error (MSE): {}\n"
              "STD of Error: {}".format(np.average(errors),
                                        round(np.std(errors), 5)))

    def gain_loss(self):
        """
            Backtest algorithm to find Success Rate, AVG gain, AVG loss

        :return: none
        """
        true_pos = 0
        events = 0
        day_change = []
        for i in range(round(len(self.predictors) * self.test_train_ratio), len(self.predictors)):
            prediction = self.predict(self.predictors[i]) + self.adjustment
            if prediction > 0.0:
                if self.response[i] > 0.0:
                    true_pos += 1
                events += 1
                day_change.append(self.response[i])
        if events == 0:
            success = 0
        else:
            success = true_pos / events
        gain = [i for i in day_change if i > 0]
        loss = [i for i in day_change if i <= 0]
        if len(gain) == 0:
            gain = 0
        else:
            gain = round(np.average(gain), 5)
        if len(loss) == 0:
            loss = 0
        else:
            loss = round(np.average(loss), 5)

        print("\n-------------------------------------\n"
              "Success Rate: {}\n"
              "Avg Gain: {}\n"
              "Avg Loss: {}\n"
              "-------------------------------------".format(success,
                                                             gain,
                                                             loss))


if __name__ == "__main__":
    data = pd.read_csv("../../GOOG.csv", header=0)
    data = data.iloc[:, 1:]

    # Solving by Backprop
    rbn_back = RBN()
    rbn_back.solve(data=data.iloc[:, 1:5])
    rbn_back.validate()
    rbn_back.gain_loss()
