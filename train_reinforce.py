from agent import Agent
from env import Environment


WINDOW_SIZE = 10
EPOCHS = 200
BATCH_SIZE = 50

def train_stock_model(agent, stockenv):

    for e in range(EPOCHS + 1):

        print("Episode " + str(e) + "/" + str(EPOCHS))
        state = stockenv.get_state(t=WINDOW_SIZE + 1)

        agent.inventory = []
        stockenv.reset_params()


        for t in range(WINDOW_SIZE, stockenv.data_len-1):

            action = agent.act(state)
            next_state = stockenv.get_state(t = t + 1)
            reward, profit = stockenv.step(agent, action, t)


            done = True if t == stockenv.data_len - 1 else False
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        if e % 10 == 0:
            agent.model.save("models/model_ep" + str(e))



        print('    Net Profit :  ' + str(stockenv.p.get_net_worth()))
        print('Episode Profit :  ' + str(stockenv.episode_profit))
        print('History is : ', stockenv.history)
        print('Buys : ' + str(stockenv.buy_count) +'     Sells: ' + str(stockenv.sell_count) + '\n')




env = Environment(WINDOW_SIZE, EPOCHS, BATCH_SIZE, 'AMD')
agent = Agent(WINDOW_SIZE)

train_stock_model(agent, env)



