import pandas as pd
import numpy as np
import random


class MLP(object):
    """
        Multi Layer Perceptron aka Feedforward Neural Net
        - MLR model with multiple layers between inputs and outputs
    """
    predictors = []
    response = []
    layer1a = []
    layer1b = []
    layer2 = []
    test_train_ratio = .8

    def __init__(self):
        pass

    def predict(self, p, agent):
        target = np.dot(p, agent)
        target = 1 / (1 + np.power(np.e, -target))
        # print("Prediction for {} is => {}".format(predictors, round(target, 4)))
        return target

    def full_predict(self, p):
        a1 = np.dot(p, self.layer1a)
        a2 = np.dot(p, self.layer1b)

        return np.dot([a1, a2], self.layer2)

    def validate(self):
        """
        :return: printed summary of model on full data set
        """
        errors = []
        for i in range(round(len(self.predictors) * self.test_train_ratio), len(self.predictors)):
            errors.append(self.response[i] - self.full_predict(self.predictors[i]))
        print("\n=================== Results ====================\n"
              "Mean Summed Error (MSE): {}\n"
              "STD of Error: {}".format(round(np.average(errors), 5),
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
            if self.full_predict(self.predictors[i]) > 0.0:
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

    def solve_backprop(self, data, alpha=.06):
        """
            Solving thru Backpropagation

        :param data: only open, high, low, close fields are used
        :param alpha: learning rate
        :return:
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]

        # init layers - Fully Connected (4, 2, 1)
        self.layer1a = [random.uniform(-1, 1) for i in range(len(self.predictors[0]))]
        self.layer1b = [random.uniform(-1, 1) for i in range(len(self.layer1a))]
        self.layer2 = [random.uniform(-1, 1) for i in range(2)]

        index = 0
        while index < 10:
            for i in range(len(self.predictors)):
                # Forward Pass
                pred1 = [self.predict(self.predictors[i], self.layer1a), self.predict(self.predictors[i], self.layer1b)]
                output = self.predict(pred1, self.layer2)

                # Backprop -> last output
                for j in range(len(self.layer2)):
                    self.layer2[j] -= alpha * pred1[j] * output * (1 - output) * len(self.layer2) / 2 * (output - self.response[i])

                # Backprop -> first of 1st hidden layer
                for k in range(len(self.layer1a)):
                    self.layer1a[k] -= alpha * self.predictors[i][k] * pred1[0] * (1 - pred1[0]) * len(self.layer1a) / 2 * (
                                pred1[0] - self.response[i])

                # Backprop -> first of 2nd hidden layer
                for l in range(len(self.layer1b)):
                    self.layer1b[l] -= alpha * self.predictors[i][l] * pred1[1] * (1 - pred1[1]) * len(self.layer1b) / 2 * (
                                pred1[1] - self.response[i])

            index += 1


if __name__ == "__main__":
    data = pd.read_csv("../../GOOG.csv", header=0)
    data = data.iloc[:, 1:]

    # Solving by Backprop
    mlp_back = MLP()
    mlp_back.solve_backprop(data=data.iloc[:, 1:5])
    mlp_back.validate()
    mlp_back.gain_loss()
