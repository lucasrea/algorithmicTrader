from agent import Agent
from env import Environment
import sys


WINDOW_SIZE = 10
EPOCHS = 200
BATCH_SIZE = 30
stock_symbol = sys.argv[1]

def train_stock_model(agent, stockenv):

    for e in range(EPOCHS + 1):

        print("Episode " + str(e) + "/" + str(EPOCHS))
        state = stockenv.get_state(t=WINDOW_SIZE + 1)

        agent.inventory = []
        stockenv.reset_params()


        for t in range(WINDOW_SIZE, stockenv.data_len-1):

            action = agent.act(state)
            next_state = stockenv.get_state(t = t + 1)
            reward = stockenv.step(agent, action, t)


            done = True if t == stockenv.data_len - 1 else False
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

            dir = stock_symbol +'_models'
            agent.model.save(dir+ "/model_ep" + str(e))


        price = stockenv.stock.vec[len(stockenv.stock.vec) - 1]
        book, net = stockenv.p.get_net_worth(stockenv.symbol, price)
        print('    Net Profit :  ' + str(net - 100000))
        print('History is : ', stockenv.history)
        print('Buys : ' + str(stockenv.buy_count) +'     Sells: ' + str(stockenv.sell_count) + '\n')




env = Environment(WINDOW_SIZE, EPOCHS, BATCH_SIZE, stock_symbol)
agent = Agent(WINDOW_SIZE, stock_symbol)

train_stock_model(agent, env)



