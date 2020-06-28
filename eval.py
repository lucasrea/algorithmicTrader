import trader
from env import Environment
from agent import Agent
import datetime
import sys

WINDOW_SIZE = 10
EPOCHS = 200
BATCH_SIZE = 95
MODEL_NAME = 'model_ep199'

stock_name = sys.argv[1]


def run_trader():

    p = trader.Portfolio('lucas')
    a = Agent(WINDOW_SIZE, is_eval=True, model_name=MODEL_NAME)
    e = Environment(WINDOW_SIZE, EPOCHS, BATCH_SIZE, stock_name, train=False)


    t_330pm = datetime.time(hour=15, minute=30)


    while True:
        now = datetime.datetime.now()
        now = now.time()

        if now == t_330pm and trader.is_market_open(): # Perform the action near the end of the day

            state = e.get_state()
            action = a.act(state)

            if action == 1: # BUY
                p.place_buy_order('AMD', e.stock.vec[-1])
            elif action == 2: # SELL
                p.place_sell_order('AMD', e.stock.vec[-1])



# Main program that will use the trained model to trade
if __name__ == "__main__":

    run_trader()
