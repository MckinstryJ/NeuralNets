import pandas as pd
import numpy as np
import random


class MLR(object):
    b = []
    predictors = []
    response = []
    test_train_ratio = .8

    def __init__(self):
        pass

    def solve_by_Least(self, data):
        """
            Solving by Least Squares

        :param data: all data... transformed to predictor / response data
        :return: none
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]

        self.b = np.dot(np.dot(np.power(np.dot(self.predictors.T,
                                               self.predictors), -1),
                               self.predictors.T),
                        self.response)

    def solve_by_Iter(self, data):
        """
            Solving thru Iteration
                For each input:
                    apply learning rate to each feature till min converages
        :param data: only open, high, low, close fields are used
        :return:
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2: ]
        self.b = [0.5 for i in range(len(self.predictors[0]))]
        for epochs in range(100):
            for i in range(len(self.predictors)):
                for j in range(len(self.predictors[0])):
                    self.b[j] = 1 / self.b[j] * 0.05 * (self.predict(self.predictors[i]) - self.response[i])

    def population(self, pop, pop_size=200):
        """
            Create Randomized Population
        :param pop: population
        :param pop_size: size of desired population
        :return: prev gen + randomized population
        """
        for i in range(pop_size):
            b = []
            for j in range(len(self.predictors[0])):
                b.append(random.uniform(-1, 1))
            pop.append(b)
        return pop

    def fitness_summary(self, pop, g, fit, top):
        print("----------------------------------------------------------------------------------")
        print("--------- Generation #{} - TOP {} AGENTS -----------------------------------------".format(g, top))
        print("----------------------------------------------------------------------------------")
        for i in range(len(pop)):
            print("--- AGENT #{} ---> FITNESS {} == {}".format(i+1, round(fit[i], 2), pop[i]))
        print()

    def selection(self, pop, g, top=10):
        """
            Selecting Top - lowest Abs Summed Error
        :param pop: complete population
        :param g: generation number
        :param top: number of top lowest
        :return: Top from pop
        """
        fitness = []
        for i in range(len(pop)):
            ase = 0
            self.b = pop[i]
            for j in range(len(self.predictors)):
                ase += np.abs(self.predict(self.predictors[j]) - self.response[j])
            fitness.append(ase)
        fit = [fitness[j] for j in sorted(range(len(fitness)), key=lambda i: fitness[i])[:top]]
        pop = [pop[j] for j in sorted(range(len(fitness)), key=lambda i: fitness[i])[:top]]
        self.fitness_summary(pop, g+1, fit, top)
        return pop

    def crossover(self, pop):
        """
            Creating Children from Top agents
        :param pop: Top Agents
        :return: Top agents and their children
        """
        for i in range(len(pop)):
            for j in range(i+1, len(pop)):
                b = []
                for k in range(len(pop[0])):
                    if random.randint(0, 1) == 0:
                        b.append(pop[i][k])
                    else:
                        b.append(pop[j][k])
            pop.append(b)

        return pop

    def mutation(self, pop, n=5, mutate_prob=.2):
        """
            Randomly adjust N elements
        :param pop: Top + children
        :param n: number of elements to be randomly adjusted if needed
        :param mutate_prob: likelihood of element changing
        :return: same and input with a possible mutation
        """
        num = 0
        for i in range(len(pop)):
            if random.random() < mutate_prob:
                for j in range(len(pop[0])):
                    if random.random() < mutate_prob and num < n:
                        pop[i][j] = random.uniform(-1, 1)
                        num += 1
        return pop

    def solve_by_Genetic(self, data):
        """
            Solving thru Genetic Algorithm
                - Instead of gradient decent... solve through specialized search
        :param data:
        :return:
        """
        self.predictors = data.iloc[:, :].pct_change().values[1:-1]
        self.response = data.iloc[:, 3].pct_change().values[2:]

        pop = []
        for generation in range(50):
            # create population
            pop = self.population(pop)
            # selection
            pop = self.selection(pop, generation)
            # crossover
            pop = self.crossover(pop)
            # mutation
            pop = self.mutation(pop)

        self.b = pop[0]

    def predict(self, p):
        target = np.dot(p, self.b)
        # print("Prediction for {} is => {}".format(predictors, round(target, 4)))
        return target

    def validate(self):
        """
        :return: printed summary of model on full data set
        """
        errors = []
        for i in range(round(len(self.predictors)*self.test_train_ratio), len(self.predictors)):
            errors.append(self.response[i] - self.predict(self.predictors[i]))
        print("\n========Results for Least====================\n"
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
        for i in range(round(len(self.predictors)*self.test_train_ratio), len(self.predictors)):
            if self.predict(self.predictors[i]) > 0.0:
                if self.response[i] >= 0.0:
                    true_pos += 1
                events += 1
                day_change.append(self.response[i])
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
    data = pd.read_csv("../../GOOG.csv", header=0)
    data = data.iloc[:, 1:]

    # Solving by Least Squares
    mlr_least = MLR()
    mlr_least.solve_by_Least(data=data.iloc[:, 1:5])
    mlr_least.predict(data.iloc[0:2, 1:5].pct_change().values[1])
    mlr_least.validate()
    mlr_least.gain_loss()

    # Solving through Iteration
    mlr_iter = MLR()
    mlr_iter.solve_by_Iter(data=data.iloc[:, 1:5])
    mlr_iter.predict(data.iloc[0:2, 1:5].pct_change().values[1])
    mlr_iter.validate()
    mlr_iter.gain_loss()

    # Solving through Genetic Algorithms
    mlr_GA = MLR()
    mlr_GA.solve_by_Genetic(data=data.iloc[:, 1:5])
    mlr_GA.validate()
    mlr_GA.gain_loss()
