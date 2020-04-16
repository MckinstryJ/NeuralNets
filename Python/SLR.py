import pandas as pd
import numpy as np


class SLR(object):
    m = 0
    b = 0
    type = None
    X = []
    y = []
    test_train_ratio = .8

    def __init__(self):
        """
        :param data: a dataframe consisting of (Date, Open, High, Low, Close, Volume,
                                            Dividend, Split, Adj_Open, Adj_High,
                                            Adj_Low, Adj_Close and Adj_Volume)
        """
        pass

    def via_Least_Squares(self, X, y, test_train_ratio = .8):
        """

        :param X: Input data aka predictor
        :param y: Target data aka response
        :param test_train_ratio: training and testing ratio
        :return: a fitted model
        """
        self.X, self.y = X, y
        self.test_train_ratio = test_train_ratio
        self.type = 0
        num, den = 0, 0
        for i in range(round(len(X)*test_train_ratio)):
            num += (X[i] - np.average(X)) * (y[i] - np.average(y))
            den += (X[i] - np.average(X))**2
        self.m = num / den
        self.b = np.average(y) - self.m * np.average(X)

        return self

    def via_Iteratively(self, X, y, test_train_ratio=0.8, epochs=10):
        """

        :param X: Input data aka predictor
        :param y: Target data aka response
        :param test_train_ratio: training and testing ratio
        :param epochs: number of rounds in dataset
        :return: a fitted model
        """
        self.X, self.y = [i - X[0] for i in X], [i - y[0] for i in y]
        self.test_train_ratio = test_train_ratio
        self.type = 1
        w = 1.0
        alpha = 0.01

        for epo in range(epochs):
            for i in range(len(X)):
                w = w + alpha * (y[i] / (w * X[i]) - 1)

        self.m = w
        self.b = np.average(y) - self.m * np.average(X)

        return self

    def model_details(self):
        """
        :return: for the SLR model as y = mx + b, returns values for m and b
        """
        if self.type == 0:
            print("\n----------SLR via Least Squares-----------\n"
                  "For model y = m * X + b:\nm = {}\nb = {}\n".format(round(self.m, 5),
                                                                      round(self.b, 5)))
        else:
            print("\n-----------SLR via Iteratively------------\n"
                  "For model y = m * X + b:\nm = {}\nb = {}\n".format(round(self.m, 5),
                                                                      round(self.b, 5)))

    def predict(self, X):
        """
        :return: a single predicted value given a specific input
        """
        return self.m * X + self.b

    def validate(self):
        """
        :return: printed summary of model on full data set
        """
        errors = []
        for i in range(round(len(self.X)*self.test_train_ratio), len(self.X)):
            errors.append(self.y[i] - self.predict(self.X[i]))
        if self.type == 0:
            print("\n========Results for Least====================\n"
                  "Mean Summed Error (MSE): {}\n"
                  "STD of Error: {}".format(round(np.average(errors), 5),
                                            round(np.std(errors), 5)))
        else:
            print("\n========Results for Iteratively==============\n"
                  "Mean Summed Error (MSE): {}\n"
                  "STD of Error: {}".format(round(np.average(errors), 5),
                                            round(np.std(errors), 5)))

    def gain_loss(self):
        true_pos = 0
        events = 0
        day_change = []
        for i in range(round(len(self.X)*self.test_train_ratio), len(self.X)):
            if self.predict(self.X[i]) / self.X[i] > 1.00:
                change = self.y[i] / self.X[i]
                if change >= 1.00:
                    true_pos += 1
                events += 1
                day_change.append(change)
        if events == 0:
            success = 0
        else:
            success = round(true_pos / events, 5)
        gain = [i for i in day_change if i > 1.00]
        loss = [i for i in day_change if i <= 1.00]
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
    data = pd.read_csv("./GOOG.csv", header=0)
    data = data.iloc[:, 1:]

    '''
        Using Least Squares
    '''
    least_ = SLR()

    least_.via_Least_Squares(X=data["Close"].iloc[:-1].values,
                             y=data["Close"].iloc[1:].values)
    least_.model_details()
    print(least_.predict(X=data["Close"].iloc[0]))
    least_.validate()
    least_.gain_loss()

    '''
        Using Iterative Method
    '''
    iter_ = SLR()
    iter_.via_Iteratively(X=data["Close"].iloc[:-1].values,
                          y=data["Close"].iloc[1:].values)
    iter_.model_details()
    print(iter_.predict(X=data["Close"].iloc[0]))
    iter_.validate()
    iter_.gain_loss()