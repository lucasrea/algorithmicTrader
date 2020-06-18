import math
import numpy as np
from get_data import GetData
from trader import Portfolio



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
        self.p = Portfolio('lucas')

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


    def take_action(self, agent, action, t):

        if action == 0:
            self.history.append('H')

        if action == 1 :  # Buy only if we already have less than 6 active buy orders
        # Having a limit on the number of buys is a more realistic way of trading

            self.buy_count += 1
            self.history.append('B')
            self.p.place_buy_order(self.symbol, self.data[t])


        elif action == 2 :  # sell

            self.sell_count += 1
            self.history.append('S')
            self.p.place_sell_order(self.symbol, self.data[t])


    def step(self, agent, action, t):
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

        worth_pre_action = self.p.get_net_worth()
        self.take_action(agent, action, t)
        worth_post_action = self.p.get_net_worth()

        profit = worth_post_action - worth_pre_action
        self.episode_profit += profit


        reward = 0.3
        if profit > 0:
            reward += math.log(profit, 2) + 1
        else:
            reward -= profit


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
        self.p.reset_info()
