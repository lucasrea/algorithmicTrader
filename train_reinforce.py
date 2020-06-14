from agent import Agent
from get_data import GetData
from env import Environment
import pandas

WINDOW_SIZE = 20
EPOCHS = 200
BATCH_SIZE = 75

def train_stock_model(agent, stockenv):


    for e in range(EPOCHS + 1):
        total_profit = 0
        print("Episode " + str(e) + "/" + str(EPOCHS))
        state = stockenv.get_state(0, WINDOW_SIZE + 1)

        agent.inventory = []
        epispde_profit = 0
        stockenv.buy_count = 0
        stockenv.sell_count = 0
        stockenv.history = []

        for t in range(stockenv.data_len-1):

            action = agent.act(state)
            #print(action)
            next_state = stockenv.get_state(t + 1, WINDOW_SIZE + 1)
            reward, profit = stockenv.determine_reward(agent, action, t)

            epispde_profit += profit

            done = True if t == stockenv.data_len - 1 else False
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("Total profit: " + str(total_profit))
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        print('    Net Profit :  ' + str(stockenv.total_profit))
        print('Episode Profit :  ' + str(epispde_profit))
        print('History is : ', stockenv.history)
        print('Buys : ' + str(stockenv.buy_count) +'     Sells: ' + str(stockenv.sell_count) + '\n')




env = Environment(WINDOW_SIZE, EPOCHS, BATCH_SIZE, 'AMD')
agent = Agent(WINDOW_SIZE)

train_stock_model(agent, env)



