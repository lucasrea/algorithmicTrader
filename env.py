import math
import numpy as np
from get_data import GetData



class Environment:

    def __init__(self, WINDOW_LENGTH, EPOCHS, BATCH_SIZE, symbol, train=True):

        """
        Constants used for things like
        - WINDOW_LENGTH : how many elements of data are in a state
        - EPOCHS : number of episodes the model is trained for
        - BATCH_SIZE : how many cycles we go until we fit the agent's model
        """
        self.WINDOW_LENGTH = WINDOW_LENGTH
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        """
        Variables for the environment data (prices of the stock)
        Train variable to determine if were being run in training mode or trading mode
        """
        self.stock = GetData(symbol, train)
        self.data = self.stock.format_data()
        self.symbol = symbol
        self.data_len = len(self.data)
        self.train = train

        """
        Parameters that are subject to reset after every training episode
        """
        self.total_profit = 0
        self.episode_profit = 0
        self.buy_count = 0
        self.sell_count = 0
        self.history = []



    def get_state(self, t=0):
        """
        Function to break the data up into window sized chunks
        Returns an n sized array up until t

        If we are in train mode, we already have all of the data, so we use the t iterator to determine where we want the state to end
        Otherwise, we need to pull the next minute of data, and retrieve the last WINDOW_LENGTH elements

        :return: a numpy array of length WINDOW_SIZE with the sigmoid transformed data
        """

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        n = self.WINDOW_LENGTH + 1

        if self.train: # A length check for when we run the live trader
            d = t - n
            block = self.data[d:t + 1] if d >= 0 else np.append(-d * [self.data[0]], self.data[0:t + 1])  # pad with t0

        else: # If we are not training, we just need to grab the last WINDOW_SIZE + 1 # of  elements
            self.data = self.stock.update_data() # Get the updated minute-by-minute data
            block = self.data[ self.data_len - n : self.data_len + 1 ]

        res = []
        for i in range(n - 1):
            res.append(sigmoid(block[i + 1] - block[i]))

        return np.array([res])


    def determine_reward(self, agent, action, t):
        """
        Function that will determine the reward for the agent depending on the action
        *** Only used during training ***

        -For a buy, we check how many better options there were (where the price was cheaper)
        -For a sell, we check if there was a better time to sell (where the price was more expensive

        :param agent:
        :param action:
        :param t: index of data
        :return: reward value, the more positive -> better
        """

        reward = 0
        profit = 0

        if action == 0:
            self.history.append('H')

        if action == 1:  # Buy only if we already have less than 6 active buy orders
        # Having a limit on the number of buys is a more realistic way of trading

            self.buy_count += 1
            self.history.append('B')
            agent.inventory.append(self.data[t])

            # Check to see if we could have bought at a better time
            min_count = 0
            for i in range(1, self.WINDOW_LENGTH - 1):
                try:
                    if self.data[t-i:t] < self.data[t]:
                        min_count += 1
                except:
                    break

            # For every better option, subtract 0.1 from a total of 1 possible point
            #reward = 0.2 - (0.02 * min_count)

            #print('Buy at ' + str(self.data[t]))
            # print(str(min_count) + ' possible buys that were better, got a reward of ' + str(reward) + ' out of 1 \n')

        elif action == 2 and len(agent.inventory) > 0:  # sell

            bought_price = min(agent.inventory)
            agent.inventory.remove(bought_price)
            reward = max(self.data[t] - bought_price, 0)
            profit = self.data[t] - bought_price

            self.total_profit += profit
            self.episode_profit += profit
            self.history.append('S')
            self.sell_count += 1


            #if profit > 0: # We made profit, it's at least a decent trade
                # If we were profitable, we add an immediate 0.7 / 1.0
            reward = 0.8 * profit

            # Check if we could have bought at a higher price
            max_count = 0
            for i in range(1, self.WINDOW_LENGTH - 1):
                try:
                    if self.data[t-i:t] < self.data[t]:
                        max_count += 1
                except:
                    break

            # For every price that was potentially higher, we subtract 0.03 from a possible 0.3 for this component
            reward += 0.2 - (0.02 * max_count)

            #print('Sold at ' + str(self.data[t]))
            # print(str(max_count) + ' possible buys that were better, got a reward of ' + str(reward) + ' out of 1 \n')
            #print('Profit of : ' + str(profit))
        return reward, profit




    def reset_params(self):
        """
        Function to reset some parameters at the begining of every episode
        :return:
        """

        self.buy_count = 0
        self.sell_count = 0
        self.history = []
        self.episode_profit = 0
