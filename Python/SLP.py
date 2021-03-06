import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


class SLP(object):
    """
        Single Layer Perceptron
        - MLR model with a final normalization function (sigmoid) before target
    """
    b = []
    predictors = []
    response = []
    max_value = 0
    test_train_ratio = .8

    def __init__(self):
        pass

    def predict(self, p, agent):
        target = np.dot(p, agent)
        target = 1 / (1 + np.power(np.e, -target))
        # print("Prediction for {} is => {}".format(predictors, round(target, 4)))
        return target

    def validate(self):
        """
        :return: printed summary of model on full data set
        """
        errors = []
        for i in range(round(len(self.predictors) * self.test_train_ratio), len(self.predictors)):
            errors.append(self.response[i] - self.predict(self.predictors[i], self.b))
        print("\n==================================== Results =====================================\n"
              "{}\n"
              "Mean Summed Error (MSE): {}\n"
              "STD of Error: {}".format(self.b,
                                        round(np.average(errors), 5),
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
            if self.predict(self.predictors[i], self.b) > 0.0:
                if self.response[i] >= 0.0:
                    true_pos += 1
                events += 1
                day_change.append((self.response[i] - .4964) * self.max_value)
        if events == 0:
            success = 0
        else:
            success = round(true_pos / events, 5)
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

    def solve_backprop(self, data, alpha=5.0):
        """
            Solving thru Backpropagation

        :param data: only open, high, low, close fields are used
        :param alpha: learning rate
        :return:
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]

        # init layers - Fully Connected (4, 1)
        self.b = [0 for i in range(len(self.predictors[0]))]

        error = []
        index, output = 0, 100
        while index < 100:
            for i in range(len(self.predictors)):
                # Forward Pass
                output = self.predict(self.predictors[i], self.b)

                # print("++++++++++++++++++++++++++++++++++++++")
                # print("With weights: {}".format(self.b))
                # print("Predicted: {}".format(output))
                # print("Actual: {}".format(self.response[i]))
                error.append(output - self.response[i])

                # Backpropagation
                for j in range(4):
                    self.b[j] -= alpha * self.predictors[i][j] * (output * (1 - output)) * .5 * (output - self.response[i])
            alpha -= .05
            if alpha < .05:
                alpha = .01
            # if index % 100 == 0:
            #     plt.plot(error)
            #     plt.show()

            index += 1
        print(np.average(error))

    def solve_iteratively(self, data, a=1.0):
        """
            Solving thru Iteration
                    For each input:
                            apply decaying learning rate to each feature till min converages
        :param data: only open, high, low, close fields are used
        :return:
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]
        self.b = [0 for i in range(len(self.predictors[0]))]

        for epoch in range(1000):
            for i in range(len(self.predictors) - 1):
                y = self.predict(self.predictors[i], self.b)
                for j in range(len(self.b)):
                    self.b[j] = self.b[j] + a * (self.response[i] - y) * self.predictors[i][j]
            a -= .05
            if a <= .05: a = .01

    def fitness(self, agent):
        error = 0
        for i in range(round(len(self.predictors) * self.test_train_ratio), len(self.predictors)):
            error += abs(self.response[i] - self.predict(self.predictors[i], agent))

        return error

    def solve_swarm(self, data, a=1.0):
        """
            Solving thru Swarm Optimization
        :param data: only open, high, low, close fields are used
        :return:
        """
        # TODO: Adjust learning rate to be more reactive to position of best agent
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]
        self.b = [0 for i in range(len(self.predictors[0]))]

        agents = [[random.uniform(-10, 10) for j in range(len(self.b))] for i in range(10)]

        best_agent = []
        best_fit = np.inf
        for epoch in range(1000):
            for agent in agents:
                if best_fit > self.fitness(agent):
                    best_fit = self.fitness(agent)
                    best_agent = agent
            for agent in agents:
                for i in range(len(self.predictors)):
                    y = self.predict(self.predictors[i], best_agent)
                    for j in range(len(self.predictors[0])):
                        agent[j] = agent[j] + a * (self.response[i] - y) * self.predictors[i][j]
            a -= .05
            if a <= .05: a = .01
        self.b = best_agent


if __name__ == "__main__":
    data = pd.read_csv("../../GOOG.csv", header=0)
    data = data.iloc[:, 1:]

    # Solving by Backprop
    slp_back = SLP()
    slp_back.solve_backprop(data=data.iloc[:, 1:5])
    slp_back.validate()
    slp_back.gain_loss()

    # Solving by Iteration
    # slp_iter = SLP()
    # slp_iter.solve_iteratively(data=data.iloc[:, 1:5])
    # slp_iter.validate()
    # slp_iter.gain_loss()

    # Solving by Swarm Optimization
    # slp_swarm = SLP()
    # slp_swarm.solve_swarm(data=data.iloc[:, 1:5])
    # slp_swarm.validate()
    # slp_swarm.gain_loss()
