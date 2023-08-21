![QuantConnect Logo](https://cdn.quantconnect.com/web/i/icon.png)
<hr>
Import Data
# QuantBook Analysis Tool 
# For more information see [https://www.quantconnect.com/docs/research/overview]
qb = QuantBook()
goog = qb.AddEquity('GOOG')
msft = qb.AddEquity('MSFT')
start_date = datetime(2017, 1, 1)
end_date = datetime(2017, 2, 1)
history = qb.History(qb.Securities.Keys, start_date, end_date, Resolution.Daily)
# history = qb.History(qb.Securities.Keys, 360, Resolution.Daily)

# Indicator Analysis
# bbdf = qb.Indicator(BollingerBands(30, 2), spy.Symbol, 360, Resolution.Daily)
# bbdf.drop('standarddeviation', 1).plot()
history = history['close'].unstack(level = 0)
history.shape
history['Spread'] = history['MSFT R735QTJ8XC9X'] - history ['GOOCV VP83T1ZUHROL']
history = history.rename(columns = {'MSFT R735QTJ8XC9X': 'MSFT','GOOCV VP83T1ZUHROL': 'GOOG'})
history
history.index = history.reset_index().loc[:, 'time'].apply(lambda x: datetime.date(x))
msft_news = qb.AddData(TiingoNews, msft.Symbol)
goog_news = qb.AddData(TiingoNews, goog.Symbol)
news = [msft_news, goog_news]
  wordScores = {
            "bad": -0.5,"good": 0.5, "negative": -0.5,
            "great": 0.5,"growth": 0.5, "fail": -0.5,
            "failed": -0.5, "success": 0.5, "nailed": 0.5,
            "beat": 0.5, "missed": -0.5, "profitable": 0.5,
            "beneficial": 0.5,"large": 0.5,"attractive": 0.5,
            "right": 0.5, "sound": 0.5,"positive": 0.5,
            "excellent": 0.5, "wrong": -0.5, "unproductive": -0.5,
            "lose": -0.5,"missing": -0.5, "mishandled": -0.5,
            "un_lucrative": -0.5, "up": 0.5, "down": -0.5,
            "unproductive": -0.5, "poor": -0.5,"wrong":-0.5,
            "worthwhile": 0.5,"lucrative": 0.5, "solid": 0.5
        }
# Define a function that computes the sentiment score of a given text based on the wordScores dictionary
def compute_sentiment_score(text, word_scores):
    words = text.lower().split(" ")
    score = sum([word_scores[word] for word in words if word in word_scores])
    return score

news_history.groupby('time').sum()
stock = news[0]
current_start = start_date
news_history = qb.History(stock.Symbol, current_start, min(end_date, current_start + timedelta(50)), Resolution.Daily)['description'].apply(lambda x: compute_sentiment_score(x, wordScores)).unstack(level = 0)
news_history = news_history.reset_index()
news_history.loc[:, 'time'] = news_history.loc[:, 'time'].apply(lambda x: datetime.date(x))
news_history = news_history.groupby('time').sum()
news_history.index.value_counts().max()
for stock in news:
    current_start = start_date
    current_end = end_date
    news_history = None
    while current_start < end_date:
        if news_history is not None:
            this_news = qb.History(stock.Symbol, current_start, min(end_date, current_start + timedelta(50)), Resolution.Daily)['description'].apply(lambda x: compute_sentiment_score(x, wordScores)).unstack(level = 0)
            this_news.index = this_news.reset_index().loc[:, 'time'].apply(lambda x: datetime.date(x))
            this_news = this_news.groupby('time').sum()
            news_history = pd.concat((news_history, this_news), axis = 0)
        else:
            news_history = qb.History(stock.Symbol, current_start, min(end_date, current_start + timedelta(50)), Resolution.Daily)['description'].apply(lambda x: compute_sentiment_score(x, wordScores)).unstack(level = 0)
            news_history.index = news_history.reset_index().loc[:, 'time'].apply(lambda x: datetime.date(x))
            news_history = news_history.groupby('time').sum()
        current_start = current_start + timedelta(51)
    history = pd.concat((history, news_history), axis = 1)
    del news_history
    del this_news
    del current_end
    del current_start
history = history.rename(columns = {"MSFT.TiingoNews R735QTJ8XC9W": "MSFT Sentiment", "GOOCV.TiingoNews VP83T1ZUHROK": "GOOG Sentiment"})
history = history.dropna()
history.shape
Create Environment
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
class StockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, history, render_mode=None):

        # Observations are dictionaries with the MSFT's and SPY's prices.
        # For this purpose, assume prices cannot go above 1,000.
        self.observation_space = gym.spaces.Dict(
            {
                "msft": gym.spaces.Box(low = 0, high = 1_000, shape = (1,), dtype=np.float32),
                "spy": gym.spaces.Box(low = 0, high = 1_000, shape = (1,), dtype=np.float32),
            }
        )

        self._msft = history.loc[:, 'MSFT']
        self._goog = history.loc[:, 'GOOG']
        self._msft_sentiment = history.loc[:, 'MSFT Sentiment']
        self._goog_sentiment = history.loc[:, 'GOOG Sentiment']

        self.date = 0

        self.portfolio = {'MSFT': 0, 'GOOG': 0, 'Cash': 100_000}
        self.cost_basis = 0
        
        # We have 3 actions, corresponding to "Long", "Hold", "Short"
        self.action_space = gym.spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the action taken.
        I.e. 0 corresponds to "Long", which will multiply our portfolio by 1.
        """
        self._action_to_direction = {
            0: 'Long',
            1: 'Hold',
            2: 'Short'
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"msft": self._msft.iloc[self.date], "goog": self._goog.iloc[self.date]}

    def _get_info(self):
        return {"msft": self._msft_sentiment.iloc[self.date], "goog": self._goog_sentiment.iloc[self.date]}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.date = 0
        observation = self._get_obs()
        self.portfolio = {'MSFT': 0, 'GOOG': 0, 'Cash': 100_000}
        self.cost_basis = 0

        return observation

    def step(self, action):
        observation = self._get_obs()
        info = self._get_info()

        fees = 0
        direction = self._action_to_direction[action]
        if direction == 'Hold':
            pass
        elif direction == 'Long':
            if self.portfolio['MSFT'] > 0:
                pass
            else:
                self.portfolio['Cash'] += self.portfolio['MSFT'] * observation['msft'] + self.portfolio['GOOG'] * observation['goog'] - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
                fees += - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
                self.portfolio['MSFT'] = 1 / (1 + .82) * self.portfolio['Cash'] // observation['msft'] 
                self.portfolio['GOOG'] = - .82 / (1 + .82) * self.portfolio['Cash'] // observation['goog']
                self.cost_basis = self.portfolio["Cash"]
                self.portfolio["Cash"] += self.portfolio['MSFT'] * observation['msft'] + self.portfolio['GOOG'] * observation['goog'] - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
                fees += - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
        elif direction == "Short":
            if self.portfolio['MSFT'] < 0:
                pass
            else:
                self.portfolio['Cash'] += self.portfolio['MSFT'] * observation['msft'] + self.portfolio['GOOG'] * observation['goog'] - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
                fees += - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
                self.portfolio['MSFT'] = - 1 / (1 + .82) * self.portfolio['Cash'] // observation['msft']
                self.portfolio['GOOG'] = .82 / (1 + .82) *self.portfolio['Cash'] // observation['goog']
                self.cost_basis = self.portfolio["Cash"]
                self.portfolio["Cash"] += self.portfolio['MSFT'] * observation['msft'] + self.portfolio['GOOG'] * observation['goog'] - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
                fees += - .005 * (self.portfolio['GOOG'] + self.portfolio['MSFT'])
            
        self.date += 1
        
        # An episode is done if the agent has reached the target
        terminated = self.date > self._msft.shape[0] - 3

        next_observation = self._get_obs()
        reward = self.portfolio['MSFT'] * (next_observation['msft']  - observation ['msft']) + self.portfolio['GOOG'] * (next_observation['goog'] - observation['goog']) + fees
        
        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, info
env = StockEnv(history)
import warnings
warnings.filterwarnings('ignore')

def run_pass(predictions, epsilon):
    state = env.reset()
    record = {}
    for i in range(500):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = predictions[i]
        n_state, reward, done, info = env.step(action)
        record[i] = {'MSFT Price': n_state['msft'], 'GOOG Price': n_state['goog'], 'MSFT Sentiment': info['msft'], 'GOOG Sentiment': info['goog'], 'reward': reward, 'action': action}
        if done:
            break
    return pd.DataFrame(record).T

import warnings
warnings.filterwarnings('ignore')

def run_pass_w_warmup(predictions, epsilon):
    state = env.reset()
    record = {}
    for i in range(500):
        if i >= sequence_length - 1 and i <= predictions.shape[0]:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = predictions[i - sequence_length + 1]
            n_state, reward, done, info = env.step(action)
            record[i] = {'MSFT Price': n_state['msft'], 'GOOG Price': n_state['goog'], 'Portfolio-MSFT': info['MSFT'], 'Portfolio-GOOG': info['GOOG'], 'Portfolio-Cash': info['Cash'], 'reward': reward, 'action': action}
        else:
            action = 1
            n_state, reward, done, info = env.step(action)
        if done:
            break
    return pd.DataFrame(record).T
# random_pass = pd.DataFrame(record).T#.loc[:, 'reward']#.plot()
# random_pass
# env.portfolio
# model = A2C('MultiInputPolicy', env)
# model.learn(total_timesteps = 500)
# obs = env.reset()
# model.predict(obs)
Create First Pass
this_pass = run_pass(None, 1000)
ideal_action = ((this_pass['MSFT Price'] - this_pass['GOOG Price']).shift(-1) - (this_pass['MSFT Price'] - this_pass['GOOG Price'])).apply(lambda x: 0 if x > 0 else 2).to_numpy()
Learn with MLP
from sklearn.neural_network import MLPRegressor
bootstrap_mean = run_pass(None, 1000).mean()
bootstrap_std = run_pass(None, 1000).std()
for epochs in range(100):
    bootstrap_mean = pd.concat((bootstrap_mean, run_pass(None, 1000).mean()), axis = 1)
    bootstrap_std = pd.concat((bootstrap_std, run_pass(None, 1000).std()), axis = 1)
# bootstrap = (bootstrap_mean.mean(axis = 1)[['MSFT Price', 'GOOG Price', 'action']], bootstrap_std.mean(axis = 1)[['MSFT Price', 'SPY Price', 'action']])
bootstrap = (bootstrap_mean.mean(axis = 1)[['MSFT Price', 'GOOG Price', "MSFT Sentiment", "GOOG Sentiment"]], bootstrap_std.mean(axis = 1)[['MSFT Price', 'GOOG Price', "MSFT Sentiment", "GOOG Sentiment"]])
bootstrap[0].to_numpy().reshape(1, -1)
# q = MLPRegressor(learning_rate_init = .01)
# q_target = MLPRegressor(learning_rate_init = .01)

# init = True
# this_pass = run_pass(None, 40)
# average_output = np.zeros((this_pass.shape[0], 3))
# counter = np.ones((this_pass.shape[0], 3))
# for epochs in range(100):
#     this_output = np.zeros((this_pass.shape[0], 3))
#     this_pass = run_pass(q.predict(x)[:, :3].argmax(1) if not init else ideal_action, 20 / (epochs + 1))
#     x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']]
#     mean = bootstrap[0]
#     std = bootstrap[1]
#     x = (x - mean) / std
#     for index in this_pass.index:
#         this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
#         this_counter = np.zeros((this_pass.shape[0], 3))
#         this_counter[index, int(this_pass.loc[index, 'action'])] = 1
#         counter = counter + this_counter
#     average_output = ((counter - 1) * average_output + this_output) / counter
    
#     if not init:
#         q_target.coefs_ = q.coefs_
#     else:
#         y_init = 100 * np.random.rand(this_pass.shape[0] - 1, 3 + x.shape[1])
#         q_target.partial_fit(x[:-1], y_init)
#         init = False
#     y_target = np.concatenate(((average_output + .000009 * q_target.predict(q_target.predict(x)[:, 3:])[:, :3].max(1).reshape(-1, 1) / 1.0003)[:-1], x.to_numpy()[1:]), axis = 1)
#     for i in range(5):
#         q.partial_fit(x[:-1], y_target)
# q = MLPRegressor(learning_rate_init = .01)
# q_target = MLPRegressor(learning_rate_init = .01)

# init = True
# this_pass = run_pass(None, 40)
# average_output = np.zeros((this_pass.shape[0], 3))
# counter = np.zeros((this_pass.shape[0], 3))
# for epochs in range(20):
#     this_output = np.zeros((this_pass.shape[0], 3))
#     this_pass = run_pass(None, 40 / (epochs + 1))
#     x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'action']]
#     mean = bootstrap[0]
#     std = bootstrap[1]
#     x = (x - mean) / std
#     for index in this_pass.index:
#         this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
#         this_counter = np.zeros((this_pass.shape[0], 3))
#         this_counter[index, int(this_pass.loc[index, 'action'])] = 1
#     counter = counter + this_counter
#     average_output = (np.where(counter == 0, 0, counter - 1) * average_output + this_output) / np.where(counter == 0, 1, counter)
    
#     sequence_length = 15
#     batched = np.stack((x[:sequence_length], x[1:1 + sequence_length]))
#     for i in range(2, x.shape[0] - sequence_length):
#         batched = np.concatenate((batched, x[i: i + sequence_length].to_numpy().reshape(1, sequence_length, x.shape[1])), axis = 0)

#     flattened_batched = batched.reshape(batched.shape[0], -1)

#     predicted_batches = batched.copy()
    
#     if not init:
#         q_target.coefs_ = q.coefs_
#     else:
#         y_init = 100 * np.random.rand(flattened_batched.shape[0], 3 + x.shape[1])
#         q_target.partial_fit(flattened_batched, y_init)
#         init = False

    
#     predicted_batches[1:, -1, :] = q_target.predict(flattened_batched)[:-1, 3:]

#     y_target = np.concatenate(((average_output[sequence_length: -1] + .9 * q_target.predict(predicted_batches.reshape(batched.shape[0], -1))[1:].max(1).reshape(-1, 1) / 1.0003), x.to_numpy()[sequence_length + 1:]), axis = 1)


    
#     # # y_target = np.concatenate(((average_output + .000009 * q_target.predict(q_target.predict(x)[:, 3:])[:, :3].max(1).reshape(-1, 1) / 1.0003)[:-1], x.to_numpy()[1:]), axis = 1)
#     for i in range(5):
#         q.partial_fit(flattened_batched[:-1], y_target)
q = MLPRegressor(learning_rate_init = .01)
q_target = MLPRegressor(learning_rate_init = .01)

init = True
this_pass = run_pass(None, 40)
average_output = np.zeros((this_pass.shape[0], 3))
counter = np.zeros((this_pass.shape[0], 3))
for epochs in range(150):
    this_output = np.zeros((this_pass.shape[0], 3))
    this_pass = run_pass(np.concatenate((np.ones(15), q.predict(flattened_batched)[:, :3].argmax(1))) if not init else np.random.randint(0, 2, 500), 50 / (epochs + 1))
    x = this_pass.loc[:, ['MSFT Price', 'GOOG Price', "MSFT Sentiment", "GOOG Sentiment"]]
    mean = bootstrap[0]
    std = bootstrap[1]
    x = (x - mean) / std
    for index in this_pass.index:
        this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
        this_counter = np.zeros((this_pass.shape[0], 3))
        this_counter[index, int(this_pass.loc[index, 'action'])] = 1
    counter = counter + this_counter
    average_output = (np.where(counter == 0, 0, counter - 1) * average_output + this_output) / np.where(counter == 0, 1, counter)
    
    sequence_length = 15
    batched = np.stack((x[:sequence_length], x[1:1 + sequence_length]))
    for i in range(2, x.shape[0] - sequence_length):
        batched = np.concatenate((batched, x[i: i + sequence_length].to_numpy().reshape(1, sequence_length, x.shape[1])), axis = 0)

    flattened_batched = batched.reshape(batched.shape[0], -1)

    predicted_batches = batched.copy()
    
    if not init:
        q_target.coefs_ = q.coefs_
    else:
        y_init = 100 * np.random.rand(flattened_batched.shape[0], 3 + x.shape[1])
        q_target.partial_fit(flattened_batched, y_init)
        init = False

    
    predicted_batches[1:, -1, :] = q_target.predict(flattened_batched)[:-1, 3:]

    y_target = np.concatenate(((average_output[sequence_length: -1] + .9 * q_target.predict(predicted_batches.reshape(batched.shape[0], -1))[1:].max(1).reshape(-1, 1) / 1.0003), x.to_numpy()[sequence_length + 1:]), axis = 1)


    
    # # y_target = np.concatenate(((average_output + .000009 * q_target.predict(q_target.predict(x)[:, 3:])[:, :3].max(1).reshape(-1, 1) / 1.0003)[:-1], x.to_numpy()[1:]), axis = 1)
    for i in range(5):
        q.partial_fit(flattened_batched[:-1], y_target)
q.score(flattened_batched[:-1], y_target)
q.predict(flattened_batched[:-1])[:, :3].argmax(1)
pd.DataFrame(q.loss_curve_).plot()
run_pass(np.concatenate((np.ones(15), q.predict(flattened_batched)[:, :3].argmax(1))), 0)['reward'].cumsum().plot()
flattened_batched2 = flattened_batched.copy()
y_target2 = y_target.copy()
from sklearn.metrics import r2_score

r2_score(y_target, q.predict(flattened_batched[:-1]))
from sklearn.metrics import mean_squared_error as mse

mse(y_target, q.predict(flattened_batched[:-1]))
pd.DataFrame(q.loss_curve_).plot()
run_pass(q.predict(x).argmax(1), 0)['reward'].cumsum().plot()
this_pass[(env._msft == env._msft['2022-03-22'])[:358].reset_index()['MSFT']]
q.predict(np.array([[ 0.78544583, 0.99811195, 0.04581817, -0.03820983, 1.68437996]]))[:, :3].argmax(1)
mean2 = mean.to_numpy().reshape(1, -1)
std2 = std.to_numpy().reshape(1, -1)
q.predict(((this_pass.loc[93, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']] - mean) / std).to_numpy().reshape(1, -1))[:, :3].argmax(1)
Learn Q-table
init = False
this_pass = run_pass(None, 40)
all_outputs = np.zeros((this_pass.shape[0], 3))
average_output = np.zeros((this_pass.shape[0], 3))
counter = np.ones((this_pass.shape[0], 3))
for epochs in range(10):
    this_output = np.zeros((this_pass.shape[0], 3))
    this_pass = run_pass(average_output.argmax(1), 100000 / (epochs + 1))
    for index in this_pass.index:
        this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
        this_counter = np.zeros((this_pass.shape[0], 3))
        this_counter[index, int(this_pass.loc[index, 'action'])] = 1
        counter = counter + this_counter
    average_output = ((counter - 1) * average_output + this_output) / counter
    # if epochs > 0:
    #     if len(all_outputs.shape) > 2:
    #         all_outputs = np.concatenate((all_outputs, this_output.reshape(-1, this_output.shape[0], this_output.shape[1])), axis = 0)
    #     else:
    #         all_outputs = np.stack((this_output, all_outputs))
    # else:
    #     all_outputs = this_output
(average_output.argmax(1) == ideal_action).sum(), len(ideal_action)
q_new = MLPRegressor(learning_rate_init = .01, solver = 'lbfgs', max_iter=10000)
q_new.fit(((x - x.mean())/x.std()), average_output)
q_new.score(((x - x.mean())/x.std()), average_output)
run_pass(q_new.predict((x - x.mean())/x.std()).argmax(1), 0)['reward'].cumsum().plot()
# pd.DataFrame(q_new.loss_curve_).plot()
average_45_day_return = pd.DataFrame(average_output).shift(45).dropna().to_numpy() + pd.DataFrame(average_output).mean(1).rolling(45).sum().dropna().to_numpy()[1:].reshape(-1, 1)

a = ((x - x.mean())/x.std())[:-44]
x = this_pass.loc[:, ['MSFT Price', "SPY Price", "Portfolio-MSFT", "Portfolio-SPY", 'Portfolio-Cash']]
x[:-45].shape, average_45_day_return.shape
q_new2 = MLPRegressor(learning_rate_init=.01, solver = 'lbfgs', max_iter= 10000)
q_new2.fit(a, average_45_day_return)
q_new2.score(a, average_45_day_return)
run_pass(q_new2.predict((x - x.mean())/x.std()).argmax(1), 0)['reward'].cumsum().plot()
from sklearn.linear_model import LinearRegression
q_lin = LinearRegression()
q_lin.fit(x[:-45], average_45_day_return)
q_lin.score(x[:-45], average_45_day_return)
(q_lin.predict(x).argmax(1) == ideal_action).sum()
run_pass(average_output.argmax(1), 0)['reward'].cumsum().plot()
run_pass(q_lin.predict(x).argmax(1), 0)['reward'].cumsum().plot()
supervised_strategy = (x['MSFT Price'] - x['SPY Price']).apply(lambda x: 2 if x > -126 else x).apply(lambda x: 0 if x < -148 else x)
(x['MSFT Price'] - x['SPY Price'])[(supervised_strategy - q_lin.predict(x).argmax(1)) == 2.]
Learn with Linear Regression
# from sklearn.linear_model import LinearRegression

# q = LinearRegression()
# q_target = LinearRegression()

# scores = []
# init = True
# this_pass = run_pass(None, 40)
# average_output = np.zeros((this_pass.shape[0], 3))
# counter = np.ones((this_pass.shape[0], 3))
# for epochs in range(10000):
#     this_output = np.zeros((this_pass.shape[0], 3))
#     this_pass = run_pass(q.predict(x)[:, :3].argmax(1) if not init else ideal_action, 20 / (epochs + 1))
#     x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']]
#     for index in this_pass.index:
#         this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
#         this_counter = np.zeros((this_pass.shape[0], 3))
#         this_counter[index, int(this_pass.loc[index, 'action'])] = 1
#         counter = counter + this_counter
#     average_output = ((counter - 1) * average_output + this_output) / counter
    
#     if not init:
#         q_target.coef_ = q.coef_
#     else:
#         y_init = 100 * np.random.rand(this_pass.shape[0] - 1, 3 + x.shape[1])
#         q_target.fit(x[:-1], y_init)
#         init = False
#     y_target = np.concatenate(((average_output + .9 * q_target.predict(q_target.predict(x)[:, 3:])[:, :3].max(1).reshape(-1, 1) / 1.0003)[:-1], x.to_numpy()[1:]), axis = 1)
#     if epochs % 1 == 0:
#         q.fit(x[:-1], y_init)

#     scores.append(q.score(x[:-1], y_init))
# pd.DataFrame({'scores': scores}).plot()
# q.score(x[:-1], y_init)
# run_pass(q.predict(x)[:, :3].argmax(1), 0)['reward'].cumsum().plot()
Learn with Normalized MLP
# q = MLPRegressor(hidden_layer_sizes=(50,))
# q_target = MLPRegressor(hidden_layer_sizes=(50,))

# init = True
# this_pass = run_pass(None, 40)
# average_output = np.zeros((this_pass.shape[0], 3))
# this_output = np.zeros((this_pass.shape[0], 3))
# counter = np.ones((this_pass.shape[0], 3))
# for epochs in range(600):
#     this_pass = run_pass(q.predict(x).argmax(1) if not init else ideal_action, 200 / (epochs + 1))
#     x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']]
#     x['MSFT Price'] = (x['MSFT Price'] - x['MSFT Price'].mean()) / x['MSFT Price'].std()
#     x['SPY Price'] = (x['SPY Price'] - x['SPY Price'].mean()) / x['SPY Price'].std()
#     for index in this_pass.index:
#         this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
#     #     this_counter = np.zeros((this_pass.shape[0], 3))
#     #     this_counter[index, int(this_pass.loc[index, 'action'])] = 1
#     #     counter = counter + this_counter
#     # average_output = ((counter - 1) * average_output + this_output) / counter
#     if not init:
#         q_target.coefs_ = q.coefs_
#     else:
#         y_init = 100 * np.random.rand(this_pass.shape[0], 3)
#         q_target.partial_fit(x, y_init)
#         init = False
#     y_target = this_output + 0.9 * np.concatenate((q_target.predict(x).max(1).reshape(-1, 1)[1:, :], np.zeros((1,1))), axis = 0) / 1.0003
#     for i in range(5):
#         q.partial_fit(x, y_target)
Learn with Random Forest/Decision Tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
q = RandomForestRegressor()
q_target = RandomForestRegressor()

init = True
this_pass = run_pass(None, 40)
average_output = np.zeros((this_pass.shape[0], 3))
counter = np.ones((this_pass.shape[0], 3))
for epochs in range(200):
    this_output = np.zeros((this_pass.shape[0], 3))
    this_pass = run_pass(q.predict(x)[:, :3].argmax(1) if not init else ideal_action, 20 / (epochs + 1))
    x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']]
    # mean = bootstrap[0]
    # std = bootstrap[1]
    # x = (x - mean) / std
    for index in this_pass.index:
        this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
        this_counter = np.zeros((this_pass.shape[0], 3))
        this_counter[index, int(this_pass.loc[index, 'action'])] = 1
        counter = counter + this_counter
    average_output = ((counter - 1) * average_output + this_output) / counter
    
    if not init:
        q_target.estimators_ = q.estimators_
    else:
        y_init = 100 * np.random.rand(this_pass.shape[0] - 1, 3 + x.shape[1])
        q_target.fit(x[:-1], y_init)
        init = False
    y_target = np.concatenate(((average_output + .9 * q_target.predict(q_target.predict(x)[:, 3:])[:, :3].max(1).reshape(-1, 1) / 1.0003)[:-1], x.to_numpy()[1:]), axis = 1)
    q.fit(x[:-1], y_target)
q.score(x[:-1], y_target)
this_pass
run_pass(q.predict(x)[:, :3].argmax(1), 0)['reward'].cumsum().plot()
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# q = DecisionTreeRegressor()

# init = True
# average_output = np.zeros((this_pass.shape[0], 3))
# this_output = np.zeros((this_pass.shape[0], 3))
# counter = np.ones((this_pass.shape[0], 3))
# for big_loop in range(400):
#     for epochs in range(10):
#         this_pass = run_pass(ideal_action if not init else None, 1 / (epochs + 1) / (big_loop + 1))
#         x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']]
#         for index in this_pass.index:
#             this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
#             this_counter = np.zeros((this_pass.shape[0], 3))
#             this_counter[index, int(this_pass.loc[index, 'action'])] = 1
#             counter = counter + this_counter
#         average_output = ((counter - 1) * average_output + this_output) / counter
#         summed_average_output = average_output[1:].sum(0).copy().reshape(1, -1)
#         for eachRow in range(2, average_output.shape[0]):
#             summed_average_output = np.concatenate((summed_average_output, average_output[eachRow:].sum(0).reshape(1, -1)))
#         summed_average_output = np.concatenate((summed_average_output, np.zeros((1, 3))))
#         y_target = average_output + .1 * summed_average_output
#         q.fit(x, y_target)
#         init = False
Learn with CNN
import torch.nn as nn
from torchvision.ops import MLP

class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            5, 8, 5, bias=True, padding=0
        )

    def forward(self, X):
        """
        X - bsz x T x dim
        returns:
           a bsz x T x dim tensor
        """
        # bsz, T, dim = X.size()

        return self.conv(X.transpose(-1, -2)).transpose(-2, -1)
q = ConvBlock(None)
q_target = ConvBlock(None)
loss_fn = nn.MSELoss()

from torch.optim import SGD

# Optimizers specified in the torch.optim package
optimizer = SGD(q.parameters(), lr=0.000000001, momentum=0.9)
import torch


def train_one_epoch(this_pass, average_output, counter):
    # Every data instance is an input + label pair
    # this_pass['MSFT Price'] = (this_pass['MSFT Price'] - this_pass['MSFT Price'].mean()) / this_pass['MSFT Price'].std()
    # this_pass['SPY Price'] = (this_pass['SPY Price'] - this_pass['SPY Price'].mean()) / this_pass['SPY Price'].std()
    x = this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']]
    mean = bootstrap[0]
    std = bootstrap[1]
    x = (x - mean) / std
    x = torch.from_numpy(x.to_numpy()).to(torch.float32)
    
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    this_output = np.zeros((this_pass.shape[0], 3))
    for index in this_pass.index:
        this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
        this_counter = np.zeros((this_pass.shape[0], 3))
        this_counter[index, int(this_pass.loc[index, 'action'])] = 1
        counter = counter + this_counter
    average_output = ((counter - 1) * average_output + this_output) / counter

    # Make predictions for this batch
    outputs = q(x)
    y_target = torch.cat(((torch.tensor(average_output[6:-2]) + 0.9 * q_target(q_target(x)[:, 3:])[:, :3].max(1).values.reshape(-1, 1) / 1.0003), x[8:]), axis = 1).to(torch.float32)

    # Compute the loss and its gradients
    loss = loss_fn(outputs[:-4], y_target)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    return loss, average_output, counter

    # # Gather data and report
    # running_loss += loss.item()
epoch_number = 0

EPOCHS = 100

best_vloss = 1_000_000.

average_output = np.zeros((this_pass.shape[0], 3))
counter = np.ones((this_pass.shape[0], 3))
x = torch.from_numpy(this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']].to_numpy()).to(torch.float32)
for epoch in range(EPOCHS):

    q.train(False)
    q_predictions = q(x).argmax(1).numpy()

    # Make sure gradient tracking is on, and do a pass over the data
    q.train(True)
    this_pass = run_pass(q_predictions if epoch > 1 else ideal_action, 1000 / (epoch + 1))
    avg_loss, average_output, counter = train_one_epoch(this_pass, average_output, counter)
    print(f'EPOCH {epoch_number + 1}: {avg_loss}')

    # if epoch % 5 == 0:
    #     with torch.no_grad():
    #         # We don't need gradients on to do reporting
    #         q.train(False)
    #         q_parameters = [eachParameter for eachParameter in q.parameters()]
    #         for i, eachParameter in enumerate(q_target.parameters()):
    #             eachParameter = eachParameter.copy_(q_parameters[i])
    

    epoch_number += 1
from sklearn.metrics import r2_score
# r2_score(y_target, q(x)[:-4])
q(x).shape, y_target.shape
average_output.shape, q_target(q_target(x)[:, 3:])[:, :3].max(1).values.reshape(-1, 1).shape
# torch.cat(((torch.tensor(average_output[8:]) + 0.001 * q_target(q_target(x)[:, 3:])[:, :3].max(1).values.reshape(-1, 1) / 1.0003), x[8:]), axis = 1).to(torch.float32)
counter.sum(1)
average_output2
q(x)[:, :3]
run_pass(q(x).argmax(1).numpy(), 0)['reward'].sum()
run_pass(q(x).argmax(1).numpy(), 0)['reward'].plot()
counter.sum(0)
average_output.argmax(1)
run_pass(average_output.argmax(1), 0)['reward'].cumsum().plot()
pd.DataFrame(average_output[:, 0] - average_output[:, 2]).hist(bins = 25)
(average_output.argmax(1) == ideal_action).sum()
(q.predict(x)[:, :3].argmax(1) == ideal_action).sum()
q.predict(x)[:, :3].argmax(1)
q.score(x[:-1], y_target)
run_pass(q.predict(x)[:, :3].argmax(1), 0)['reward'].plot()
pd.DataFrame(q.loss_curve_)[400:].plot()
Transformer Model
import torch.nn as nn
import torch
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        if len(x.size()) == 2:
            return self.dropout(torch.cat((x.reshape(x.size(0), -1, x.size(1)), self.pe[:x.size(0), :, :4]), axis = -1)).transpose(0, 1)
        else:
            return self.dropout(torch.cat((x, self.pe[:x.size(0), :, :4].expand(-1, x.size(1), -1)), axis = -1))   
class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, length, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp, 
            nhead=nhead,
            batch_first=True
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=nlayers, 
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ninp, 
            nhead=nhead,
            batch_first=True
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers, 
            norm=None
            )

        self.norm1 = nn.LayerNorm(
            ninp
        )

        self.hidden1= nn.Linear(
            in_features = ninp * length,
            out_features = 100
        )

        self.hidden2 = nn.Linear(
            in_features = 100,
            out_features = 5
        )

        self.linear1 = nn.Linear(
            in_features=length,
            out_features=1
            )

        self.linear2 = nn.Linear(
            in_features = ninp,
            out_features = 3
        )

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            # device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(src.size(0) if len(src.size()) == 2 else src.size(1))#.to(device)
                self.src_mask = mask

        else:
            self.src_mask = None
        # print(f"input: {src.shape}")
        src = self.pos_encoder(src)
        # print(f"pos encoded: {src.shape}")
        encoder_output = self.encoder(src, mask = self.src_mask)
        # print(f"transformer encoded: {encoder_output.shape}")
        # output = self.decoder(
        #     tgt=src,
        #     memory=encoder_output,
        #     tgt_mask=self.src_mask,
        #     memory_mask=self.src_mask
        #     )
        # output = self.norm1(src)
        # print(src.shape)
        output = self.hidden1(encoder_output.reshape(encoder_output.size(0), -1))
        # print(f"hidden layer 1: {output.shape}")
        # print(output.shape)
        output = self.hidden2(output)
        # print(f"hidden Layer 2: {output.shape}")
        # print(output.shape)
        # output = self.linear1(output.transpose(-2, -1)).transpose(-2, -1)
        # output = self.linear2(output)
        return output
import torch


def train_one_epoch(q, flattened_batched, y_target):

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    outputs = q(flattened_batched)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, y_target)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    return loss, q, outputs

    # # # Gather data and report
    # # running_loss += loss.item()

    # MLP Code Below
    
    # q.partial_fit(flattened_batched, y_target)

    # return q
def train_one_pass(this_pass, average_output, counter, q, q_target, init):


    ## Code for Torch

    this_output = torch.zeros((this_pass.shape[0], 3))
    this_counter = torch.zeros((this_pass.shape[0], 3))
    this_pass = run_pass(None if not init else ideal_action, 40)
    x = this_pass.loc[:, ['MSFT Price', 'SPY Price']]
    mean = bootstrap[0]
    std = bootstrap[1]
    x = torch.from_numpy(((x - mean) / std).to_numpy()).to(torch.float32)

    with torch.no_grad():
        for index in this_pass.index:
            this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
            this_counter[index, int(this_pass.loc[index, 'action'])] = 1
        counter += this_counter
        average_output = (torch.where(counter == 0, 0, counter - 1) * average_output + this_output) / torch.where(counter == 0, 1, counter)
    
        sequence_length = 15
        batched = torch.stack((x[:sequence_length], x[1:1 + sequence_length]))
        for i in range(2, x.shape[0] - sequence_length):
            batched = torch.cat((batched, x[i: i + sequence_length].reshape(1, sequence_length, x.shape[1])), axis = 0)

        # flattened_batched = batched.reshape(batched.size(0), -1)

        predicted_batches = batched.detach().clone()

        predicted_batches[1:, -1, :] = q_target(batched)[:-1, 3:]
        # predicted_batches[1:, -1, :] = q_target(flattened_batched)[:-1, 3:] 
    
        y_target = torch.cat(((average_output[sequence_length: -1] + .9 * q_target(predicted_batches)[1:, 3:].max(1).values.reshape(-1, 1) / 1.0003), x[sequence_length + 1:]), axis = 1)
        # y_target = torch.cat(((average_output[sequence_length: -1] + .9 * q_target(predicted_batches.reshape(batched.shape[0], -1))[1:, 3:].max(1).values.reshape(-1, 1) / 1.0003), x[sequence_length + 1:]), axis = 1)

    for i in range(5):
        # loss, q, outputs = train_one_epoch(q, flattened_batched[:-1], y_target)
        loss, q, outputs = train_one_epoch(q, batched[:-1], y_target)

    with torch.no_grad():
        q_parameters = [eachParameter for eachParameter in q.parameters()]
        for i, eachParameter in enumerate(q_target.parameters()):
            eachParameter = eachParameter.copy_(q_parameters[i])

    return loss, average_output, counter, outputs, y_target, q, q_target, init
def prep_average_output(average_output, counter):
    for passes in range(50):
        this_pass = run_pass(ideal_action, 1000)
        this_output = np.zeros((this_pass.shape[0], 3))
        this_counter = np.zeros((this_pass.shape[0], 3))
        for index in this_pass.index:
            this_output[index, int(this_pass.loc[index, 'action'])] = this_pass.loc[index, 'reward']
            this_counter[index, int(this_pass.loc[index, 'action'])] = 1
        counter = counter + this_counter
        average_output = (np.where(counter == 0, 0, counter - 1) * average_output + this_output) / np.where(counter == 0, 1, counter)
    return average_output
q = TransformerModel(15, 6, 6, 6, 2)
q_target = TransformerModel(15, 6, 6, 6, 2)

loss_fn = nn.MSELoss()

from torch.optim import SGD, Adam

# Optimizers specified in the torch.optim package
# optimizer = SGD(q.parameters(), lr=0.001, momentum=0.9)
optimizer = Adam(q.parameters(), lr=0.01)

epoch_number = 0

EPOCHS = 75

best_vloss = 1_000_000.

init = True
average_output = torch.zeros((this_pass.shape[0], 3))
counter = torch.zeros((this_pass.shape[0], 3))
# average_output = prep_average_output(average_output, counter)
this_output = torch.zeros((this_pass.shape[0], 3))
this_pass = run_pass(ideal_action, 10000)
x = torch.from_numpy(this_pass.loc[:, ['MSFT Price', 'SPY Price', 'Portfolio-MSFT', 'Portfolio-SPY', 'Portfolio-Cash']].to_numpy()).to(torch.float32)
for epoch in range(EPOCHS):

    # q.train(False)
    # q_predictions = q(x).argmax(1).numpy()

    # if not init and epoch % 5 == 0:
    #     q_target.coefs_ = q.coefs_

    # Make sure gradient tracking is on, and do a pass over the data
    q.train(True)

    this_pass = run_pass(None if epoch > EPOCHS else ideal_action, 1000 / (epoch + 1))
    avg_loss, average_output, counter, outputs, y_target, q, q_target, init = train_one_pass(this_pass, average_output, counter, q, q_target, init)
    print(f'EPOCH {epoch_number + 1}: {avg_loss}')

    # if epoch % 5 == 0:
    #     with torch.no_grad():
    #         # We don't need gradients on to do reporting
    #         q.train(False)
    #         q_parameters = [eachParameter for eachParameter in q.parameters()]
    #         for i, eachParameter in enumerate(q_target.parameters()):
    #             eachParameter = eachParameter.copy_(q_parameters[i])
    

    epoch_number += 1
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

r2_score(y_target.detach().numpy(), outputs.detach().numpy())
with torch.no_grad():
    print(r2_score(y_target, outputs))
# mse(y_target.detach().numpy(), outputs[0].detach().numpy())
batched[0]
(this_pass.loc[:14, ['MSFT Price', 'SPY Price']] - bootstrap[0]) / bootstrap[1]
this_pass.loc[14]
run_pass_w_warmup(outputs[:, :3].argmax(1).numpy(), 0)['reward'].cumsum().plot()
outputs[:, :3].argmax(1).numpy()
from sklearn.metrics import r2_score
with torch.no_grad():
    print(r2_score(outputs, y_target))
# outputs.shape, y_target.shape
# average_output.shape, q_target(q_target(x)[:, 3:])[:, :3].max(1).values.reshape(-1, 1).shape
Causal Self-Attention Model
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mod_dim % config.nhead == 0
        self.pos_encoder = PositionalEncoding(config.mod_dim, config.dropout)
        self.attn_lin = nn.Linear(  # computes key, query, value projections for all heads, but in batch
            config.mod_dim, 3 * config.mod_dim, bias=config.bias
        )
        self.out_lin = nn.Linear(  # projection to re-combine heads' values
            config.mod_dim, config.mod_dim, bias=config.bias
        )
        self.out_lin2 = nn.Linear(config.context_length, 1)
        self.out_lin3 = nn.Linear(config.mod_dim, 3)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.mod_dim = config.mod_dim
        mask = ~torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "mask", mask.unsqueeze(0).unsqueeze(0)
        )  # 1 x 1 x len x len

    def get_queries_keys_values(self, X):
        """
        X - bsz x T x model_dim
        returns:
          Q, K, V, each bsz x T x nhead x model_dim/nhead
        """
        bsz, T, dim = X.size()

        # calculate query, key, values for all heads in parallel
        Q, K, V = self.attn_lin(
            X
        ).split(  # splits into 3 bsz x T x mod_dim matrices, along 3rd dim
            self.mod_dim, dim=2
        )

        # form bsz x T x nhead x dim/nhead -> bsz x nhead x T x dim/nhead views of each matrix
        Q = Q.view(bsz, T, self.nhead, self.mod_dim // self.nhead).transpose(1, 2)
        K = K.view(bsz, T, self.nhead, self.mod_dim // self.nhead).transpose(1, 2)
        V = V.view(bsz, T, self.nhead, self.mod_dim // self.nhead).transpose(1, 2)
        return Q, K, V

    def compute_causal_attn_matrices(self, Q, K):
        """
        Q - bsz x nhead x T x head_dim
        K - bsz x nhead x T x head_dim
        returns:
          bsz x nhead x T x T attention tensor
        """
        att = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(
            self.mod_dim
        )  # bsz x nhead x T x T
        if self.mask.size(2) == att.size(2) and self.mask.size(3) == att.size(
            3
        ):  # Confirm mask is the same size as att, it will not when last batch isn't full
            att = att.masked_fill(self.mask, float("-inf"))
        else:
            att = att.masked_fill(
                self.mask[:, :, : att.size(2), : att.size(3)], float("-inf")
            )
        att = self.attn_dropout(att)  # bsz x nhead x T x T
        return att

    def forward(self, X):
        """
        X - bsz x T x model_dim
        returns:
          bsz x T x V logits tensor
        """
        X = self.pos_encoder(X)
        bsz, T, dim = X.size()  # batch size, sequence length, input embedding dim
        Q, K, V = self.get_queries_keys_values(X)
        att = self.compute_causal_attn_matrices(Q, K)  # bsz x nhead x T x T
        Z = self.out_lin(torch.matmul(att, V).transpose(1, 2).reshape(bsz, T, -1))
        Z = self.out_lin2(Z.transpose(0, 2).transpose(0,1))
        Z = self.out_lin3(Z.transpose(-1, -2))
        return self.out_dropout(Z)
class Config():
    def __init__(self):
        self.context_length = 358
        self.vocab_size: int = 3
        self.nlayer: int = 3
        self.nhead: int = 10
        self.mod_dim: int = 10
        self.dropout = 0.1
        self.bias: bool = False
        self.kW: int = 5
config = Config()
q = CausalSelfAttention(config)
batched[:50].shape
q(x)[3:]
batched
