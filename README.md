# algorithmicTrader
This is my first project on algorithmic trading with the use of reinforcement learning. I've designed this as an automated function that could be run on the cloud and simulate actual trading. Though this project doesn't account for transaction costs and price increases/decrases when buying/selling, it does a decent job at approximating real time trading.


The general flow of this program is as follows.
First you will want to train a model with a particular stock. For example, AMD. This will begin the training process of the model. It may take some time. The environment settings are in the train_reinforce.py file. So if you want, you can change things like, WINDOW_SIZE (amount of data in a state), EPOCHS (or episode count, how many iterations the training process will do), and BATCH_SIZE (how many states pass before the model is fitted).

```
python train_reinforce.py AMD
```
Models are saved in a directory specified by the stock name you give (ex : amd_models). 

Transactions are done through the trader.py class called Portfolio. There are functions that place buy and sell orders based on the current value of cash or book value of stock (ex : buying AMD shares that evaluate to 1% of our current cash or selling 1% of AMD shares based on how much we already own). You can change the amount of stock you buy with the parameter, self.PCT_OF_MAX.

One episode should produce something similar to the following
```
Episode 0/200
    Net Profit :  28433.811742782593
History is :  ['H', 'H', 'B', 'H', 'S', 'H', 'H', 'H', 'S', 'S', 'S', 'H', 'B', 'B', 'S', 'H', 'H', 'B', 'H', 'S', 'H', 'H', 'S', 'H', 'H', 'S', 'B', 'S', 'B', 'H', 'S', 'B', 'S', 'H', 'B', 'S', 'H', 'S', 'B', 'B', 'B', 'B', 'H', 'B', 'H', 'S', 'S', 'S', 'S', 'S', 'B', 'H', 'B', 'B', 'B', 'S', 'S', 'H', 'H', 'S', 'B', 'B', 'S', 'H', 'S', 'S', 'H', 'H', 'H', 'H', 'H', 'B', 'S', 'S', 'H', 'S', 'B', 'S', 'B', 'S', 'S', 'B', 'H', 'S', 'S', 'S', 'B', 'B', 'S', 'B', 'S', 'S', 'B', 'H', 'S', 'S', 'S', 'S', 'S', 'S', 'H', 'B', 'H', 'B', 'S', 'B', 'H', 'B', 'S', 'B', 'B', 'B', 'H', 'S', 'B', 'S', 'B', 'B', 'B', 'B', 'S', 'S', 'S', 'B', 'B', 'B', 'B', 'B', 'S', 'S', 'B', 'H', 'B', 'B', 'B', 'H', 'B', 'H', 'S', 'S', 'S', 'S', 'S', 'H', 'S', 'S', 'S', 'S', 'H', 'S', 'B', 'S', 'B', 'B', 'S', 'S', 'S', 'S', 'B', 'H', 'S', 'B', 'S', 'H', 'S', 'S', 'H', 'S', 'S', 'S', 'B', 'S', 'B', 'S', 'S', 'H', 'H', 'B', 'S', 'B', 'B', 'S', 'S', 'S', 'B', 'B', 'H', 'H', 'B', 'B', 'S', 'B', 'B', 'B', 'H', 'B', 'H', 'S', 'B', 'S', 'S', 'H', 'S', 'B', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'B', 'H', 'H', 'S', 'S', 'S', 'B', 'S', 'S', 'S', 'H', 'S', 'S', 'B', 'B', 'S', 'B', 'S', 'S', 'S', 'S', 'B', 'S', 'S']
Buys : 75     Sells: 112
```
This indicates that on the 0th episode, we made ~ $28,433, given a starting portfolio value of $100,000 (not bad). We also see the history of actions the model took as well as the buy and sell count. 

Throughout the training process you will have some models that produce very good results (for AMD I had a model that gave me almost an 80% return over 1 year). You will also see some models that dont perform well, some not even buying or selling. I'm not sure why that is, as this is my first project using reinforcmenet learning so my understanding on the subject matter is limited. However, as the episodes go on, you will notice incrementally better results - an indication that the training process is working.


The next part of the porject is actually using one of the models to run this on real, live data. The models produced seem to work well in the short term (buying and selling after a few days) so it's a little hard to show those results. But the way to deploy the model in real-time is by doing.
```
python eval.py AMD
```
The following code segment is essentially how the real-time trader works. Since during training we only buy, sell, or hold in one day, we need to reflect that here. A loop continues to run until the current time is 3:30 pm, and the market is open. At that time, we get the current state, and perform the corresponding action.

```
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
```
Make sure to reset the json file that hold your portfolio into before running in real time. That is also where you should see all your orders being filled. 
