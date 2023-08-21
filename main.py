# region imports
from AlgorithmImports import *
from QuantConnect.Data.Custom.Tiingo import *
import pickle
import gym

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor



class VirtualGreenWolf(QCAlgorithm):

    def Initialize(self):
        self.start_date = datetime(2010, 1, 1)
        self.end_date = datetime(2011, 1, 1)
        self.SetStartDate(self.start_date)  #Set Start Date
        self.SetEndDate(self.end_date)    #Set End Date
        self.SetCash(10_000_000)           #Set Strategy Cash
        self.Cash = 10_000_000
        self.ticker1 = "XOM"
        self.ticker2 = "BP"
        # self.column1 = "MSFT R735QTJ8XC9X"
        # self.column2 = "GOOCV VP83T1ZUHROL"
        # self.news_column1 = "MSFT.TiingoNews R735QTJ8XC9W"
        # self.news_column2 = "GOOCV.TiingoNews VP83T1ZUHROK"
        self.column1 = "XON R735QTJ8XC9X"
        self.column2 = "BP R735QTJ8XC9X"
        # self.column1 = "DIS R735QTJ8XC9X"	
        # self.column2 = "NFLX SEWJWLJNHZDX"
        # self.news_column1 = "DIS.TiingoNews R735QTJ8XC9W"
        # self.news_column2 =  "NFLX.TiingoNews SEWJWLJNHZDW"
        # self.column1 = "SPY R735QTJ8XC9X"
        # self.column2 = "QQQ RIWIV7K5Z9LX"	
        # self.news_column1 = "SPY.TiingoNews R735QTJ8XC9W"
        # self.news_column2 = "QQQ.TiingoNews RIWIV7K5Z9LW"
        self.news_column1 = "XON.TiingoNews R735QTJ8XC9W"
        self.news_column2 = "BP.TiingoNews R735QTJ8XC9W"
        self.beta = .82
        self.ticker1_symbol = self.AddEquity(self.ticker1, Resolution.Daily).Symbol
        self.ticker2_symbol = self.AddEquity(self.ticker2, Resolution.Daily).Symbol
        self.Long = None

        self.wordScores = {
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

        train = False

        self.model = pickle.loads(self.ObjectStore.ReadBytes(f"{self.ProjectId}/MLPModel"))
        self.mean = pickle.loads(self.ObjectStore.ReadBytes(f"{self.ProjectId}/Mean"))
        self.std = pickle.loads(self.ObjectStore.ReadBytes(f"{self.ProjectId}/Std"))
        self.qb1 = QuantBook()
        self.qb2 = QuantBook()

        if train:
            history = self.loadData()
            env = StockEnv(history, self.ticker1, self.ticker2, setCash = self.Cash, beta = self.beta)

            self.mean, self.std = self.bootstrap(env)
            self.model = self.learn_with_MLP(env, saved_q = self.model)
            save_successful = self.ObjectStore.SaveBytes(f"{self.ProjectId}/MLPModel", pickle.dumps(self.model))
            self.Log(f'Model saved: {save_successful}')
            self.mean = self.mean.to_numpy().reshape(1, -1)
            self.std = self.std.to_numpy().reshape(1, -1)
            save_mean = self.ObjectStore.SaveBytes(f"{self.ProjectId}/Mean", pickle.dumps(self.mean))
            save_std = self.ObjectStore.SaveBytes(f"{self.ProjectId}/Std", pickle.dumps(self.std))
            self.Log(f'Mean saved: {save_mean}')
            self.Log(f'Std saved: {save_std}')

        else:
            self.ticker1_news = self.qb2.AddData(TiingoNews, self.ticker1_symbol, Resolution.Daily)
            self.ticker2_news = self.qb2.AddData(TiingoNews, self.ticker2_symbol, Resolution.Daily)
            # self.ticker1_news = self.AddData(TiingoNews, self.ticker1_symbol, Resolution.Daily)
            # self.ticker2_news = self.AddData(TiingoNews, self.ticker2_symbol, Resolution.Daily)

        self.ticker1_window = RollingWindow[float](15)
        self.ticker2_window = RollingWindow[float](15)
        self.ticker1_sentiment = RollingWindow[float](15)
        self.ticker2_sentiment = RollingWindow[float](15)
        self.SetWarmup(timedelta(45))
        

    def OnData(self, data: Slice):
        # if self.Portfolio.TotalPortfolioValue < .93 * self.Cash:
        #     self.Liquidate()
        #     return
        if data.ContainsKey(self.ticker1_symbol):
            self.ticker1_window.Add(self.Portfolio[self.ticker1].Price)
            self.ticker2_window.Add(self.Portfolio[self.ticker2].Price)
            try:
                # this_news = self.qb2.History(self.ticker1_news.Symbol, self.Time - timedelta(1), self.Time, Resolution.Daily)['description'].apply(lambda x: self.compute_sentiment_score(x, self.wordScores)).sum()
                # # this_news = self.History(self.ticker1_news.Symbol, self.Time - timedelta(1), self.Time, Resolution.Daily)['description'].apply(lambda x: self.compute_sentiment_score(x, self.wordScores)).sum()
                # self.ticker1_sentiment.Add(this_news.sum())
                self.ticker1_sentiment.Add(0)
            except:
                pass
            try:
                # this_news = self.qb2.History(self.ticker2_news.Symbol, self.Time - timedelta(1), self.Time, Resolution.Daily)['description'].apply(lambda x: self.compute_sentiment_score(x, self.wordScores)).sum()
                # # this_news = self.History(self.ticker2_news.Symbol, self.Time - timedelta(1), self.Time, Resolution.Daily)['description'].apply(lambda x: self.compute_sentiment_score(x, self.wordScores)).sum()
                # self.ticker2_sentiment.Add(this_news.sum())
                self.ticker2_sentiment.Add(0)
            except:
                pass
            if not self.ticker1_window.IsReady or not self.ticker2_window.IsReady or not self.ticker1_sentiment.IsReady or not self.ticker2_sentiment.IsReady:
                return
            x = np.array([list(self.ticker1_window)[::-1], list(self.ticker2_window)[::-1], list(self.ticker1_sentiment)[::-1], list(self.ticker2_sentiment)[::-1]]).transpose()
            direction = self.model.predict(((x - self.mean) / self.std).reshape(1, -1))[:, :3].argmax(1)
            self.Log(f"{self.Time}: {direction}")
            if direction == 0:
                if self.Portfolio[self.ticker1].Quantity > 0:
                    pass
                else:
                    self.SetHoldings(self.ticker1, .5 * 1 / (1 + self.beta))
                    self.SetHoldings(self.ticker2, -.5 * self.beta / (1 + self.beta))
            elif direction == 2:
                if self.Portfolio[self.ticker1].Quantity < 0:
                    pass
                else:
                    self.SetHoldings(self.ticker1, -.5 * 1 / (1 + self.beta))
                    self.SetHoldings(self.ticker2, .5 * self.beta / (1 + self.beta))
           
           

    def loadData(self):
        history = self.qb1.History(self.Securities.Keys, self.start_date, self.end_date, Resolution.Daily)
        history = history['close'].unstack(level = 0)
        # history['Spread'] = history[self.ticker1.Symbol] - history [self.ticker2.Symbol]
        history = history.rename(columns = {self.column1: self.ticker1, self.column2: self.ticker2})        
        history.index = history.reset_index().loc[:, 'time'].apply(lambda x: datetime.date(x))
        history = self.addNews(history)
        return history
    
    def addNews(self, history):
        self.ticker1_news = self.qb2.AddData(TiingoNews, self.ticker1_symbol, Resolution.Daily)
        self.ticker2_news = self.qb2.AddData(TiingoNews, self.ticker2_symbol, Resolution.Daily)
        news = [self.ticker1_news, self.ticker2_news]
        for stock in news:
            current_start = self.start_date
            news_history = None
            this_news = None
            while current_start < self.end_date:
                if news_history is not None:
                    this_news = self.qb2.History(stock.Symbol, current_start, min(self.end_date, current_start + timedelta(50)), Resolution.Daily)['description'].apply(lambda x: self.compute_sentiment_score(x, self.wordScores)).unstack(level = 0)
                    this_news = this_news.reset_index()
                    this_news.loc[:, 'time'] = this_news.loc[:, 'time'].apply(lambda x: datetime.date(x))
                    this_news = this_news.groupby('time').sum()
                    news_history = pd.concat((news_history, this_news), axis = 0)
                else:
                    news_history = self.qb2.History(stock.Symbol, current_start, min(self.end_date, current_start + timedelta(50)), Resolution.Daily)['description'].apply(lambda x: self.compute_sentiment_score(x, self.wordScores)).unstack(level = 0)
                    news_history = news_history.reset_index()
                    news_history.loc[:, 'time'] = news_history.loc[:, 'time'].apply(lambda x: datetime.date(x))
                    news_history = news_history.groupby('time').sum()
                current_start = current_start + timedelta(51)
            news_history = news_history.groupby('time').mean()
            history = history.groupby('time').mean()
            history = pd.concat((history, news_history), axis = 1)
            del news_history
            del this_news
            del current_start
        history = history.rename(columns = {self.news_column1: f"{self.ticker1} Sentiment", self.news_column2: f"{self.ticker2} Sentiment"})
        history = history.dropna()
        return history

    def compute_sentiment_score(self, text, word_scores):
        words = text.lower().split(" ")
        score = sum([word_scores[word] for word in words if word in word_scores])
        return score

    def run_pass(self, predictions, epsilon, env):
        state = env.reset()
        record = {}
        for i in range(500):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = predictions[i]
            n_state, reward, done, info = env.step(action)
            record[i] = {f'{self.ticker1} Price': n_state[self.ticker1], f'{self.ticker2} Price': n_state[self.ticker2], f'{self.ticker1} Sentiment': info[self.ticker1], f'{self.ticker2} Sentiment': info[self.ticker2], 'reward': reward, 'action': action}
            if done:
                break
        return pd.DataFrame(record).T

    def learn_with_MLP(self, env, saved_q = None):
        q = MLPRegressor(learning_rate_init = .01)
        q_target = MLPRegressor(learning_rate_init = .01)

        
        init = False
        if saved_q is not None:
            q = saved_q
    
        
        this_pass = self.run_pass(None, 40, env)
        average_output = np.zeros((this_pass.shape[0], 3))
        counter = np.zeros((this_pass.shape[0], 3))
        for epochs in range(50):
            this_output = np.zeros((this_pass.shape[0], 3))
            this_pass = self.run_pass(np.concatenate((np.ones(15), q.predict(flattened_batched)[:, :3].argmax(1))) if init else np.random.randint(0, 2, 500), 25 / (epochs + 1), env)
            x = this_pass.loc[:, [f'{self.ticker1} Price', f'{self.ticker2} Price', f'{self.ticker1} Sentiment', f'{self.ticker2} Sentiment']]
            x = (x - self.mean) / self.std
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
            
            if init:
                q_target.coefs_ = q.coefs_
                q_target.intercepts_ = q.intercepts_
            else:
                y_init = 100 * np.random.rand(flattened_batched.shape[0], 3 + x.shape[1])
                q_target.partial_fit(flattened_batched, y_init)
                init = True
                if saved_q is not None:
                    q_target.coefs_ = q.coefs_
                    q_target.intercepts_ = q.intercepts_

            
            predicted_batches[1:, -1, :] = q_target.predict(flattened_batched)[:-1, 3:]

            y_target = np.concatenate(((average_output[sequence_length: -1] + .9 * q_target.predict(predicted_batches.reshape(batched.shape[0], -1))[1:].max(1).reshape(-1, 1) / 1.0003), x.to_numpy()[sequence_length + 1:]), axis = 1)

            for i in range(5):
                q.partial_fit(flattened_batched[:-1], y_target)
        
        return q



    def bootstrap(self, env):
        bootstrap_mean = self.run_pass(None, 1000, env).mean()
        bootstrap_std = self.run_pass(None, 1000, env).std()
        for epochs in range(100):
            bootstrap_mean = pd.concat((bootstrap_mean, self.run_pass(None, 1000, env).mean()), axis = 1)
            bootstrap_std = pd.concat((bootstrap_std, self.run_pass(None, 1000, env).std()), axis = 1)
        bootstrap = (bootstrap_mean.mean(axis = 1)[[f'{self.ticker1} Price', f'{self.ticker2} Price', f"{self.ticker1} Sentiment", f"{self.ticker2} Sentiment"]], bootstrap_std.mean(axis = 1)[[f'{self.ticker1} Price', f'{self.ticker2} Price', f"{self.ticker1} Sentiment", f"{self.ticker2} Sentiment"]])
        return bootstrap


class StockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, history, ticker1, ticker2, setCash = 100_000, beta = 1, render_mode=None):

        # # Observations are dictionaries with the MSFT's and SPY's prices.
        # # For this purpose, assume prices cannot go above 1,000.
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         self.ticker1: gym.spaces.Box(low = 0, high = 1_000, shape = (1,), dtype=np.float32),
        #         self.ticker2: gym.spaces.Box(low = 0, high = 1_000, shape = (1,), dtype=np.float32),
        #     }
        # )

        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.SetCash = setCash
        self.beta = beta
        self._ticker1 = history.loc[:, ticker1]
        self._ticker2 = history.loc[:, ticker2]
        self._ticker1_sentiment = history.loc[:, f'{ticker1} Sentiment']
        self._ticker2_sentiment = history.loc[:, f'{ticker2} Sentiment']

        self.date = 0

        self.portfolio = {ticker1: 0, ticker2: 0, 'Cash': self.SetCash}
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
        return {self.ticker1: self._ticker1.iloc[self.date], self.ticker2: self._ticker2.iloc[self.date]}

    def _get_info(self):
        return {self.ticker1: self._ticker1_sentiment.iloc[self.date], self.ticker2: self._ticker2_sentiment.iloc[self.date]}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.date = 0
        observation = self._get_obs()
        self.portfolio = {self.ticker1: 0, self.ticker2: 0, 'Cash': 100_000}
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
            if self.portfolio[self.ticker1] > 0:
                pass
            else:
                self.portfolio['Cash'] += self.portfolio[self.ticker1] * observation[self.ticker1] + self.portfolio[self.ticker2] * observation[self.ticker2] - .005 * (self.portfolio[self.ticker1] + self.portfolio[self.ticker2])
                fees += - .005 * (self.portfolio[self.ticker1] + self.portfolio[self.ticker2])
                self.portfolio[self.ticker1] = 1 / (1 + self.beta) * self.portfolio['Cash'] // observation[self.ticker1]
                self.portfolio[self.ticker2] = - self.beta / (1 + self.beta) * self.portfolio['Cash'] // observation[self.ticker2]
                self.cost_basis = self.portfolio["Cash"]
                self.portfolio["Cash"] += self.portfolio[self.ticker1] * observation[self.ticker1] + self.portfolio[self.ticker2] * observation[self.ticker2] - .005 * (self.portfolio[self.ticker2] + self.portfolio[self.ticker1])
                fees += - .005 * (self.portfolio[self.ticker2] + self.portfolio[self.ticker1])
        elif direction == "Short":
            if self.portfolio[self.ticker1] < 0:
                pass
            else:
                self.portfolio['Cash'] += self.portfolio[self.ticker1] * observation[self.ticker1] + self.portfolio[self.ticker2] * observation[self.ticker2] - .005 * (self.portfolio[self.ticker2] + self.portfolio[self.ticker1])
                fees += - .005 * (self.portfolio[self.ticker2] + self.portfolio[self.ticker1])
                self.portfolio[self.ticker1] = - 1 / (1 + self.beta) * self.portfolio['Cash'] / observation[self.ticker1]
                self.portfolio[self.ticker2] = self.beta / (1 + self.beta) * self.portfolio['Cash'] / observation[self.ticker2]
                self.cost_basis = self.portfolio["Cash"]
                self.portfolio["Cash"] += self.portfolio[self.ticker1] * observation[self.ticker1] + self.portfolio[self.ticker2] * observation[self.ticker2] - .005 * (self.portfolio[self.ticker2] + self.portfolio[self.ticker1])
                fees += - .005 * (self.portfolio[self.ticker2] + self.portfolio[self.ticker1])
            
        self.date += 1
        
        # An episode is done if the agent has reached the target
        terminated = self.date > self._ticker1.shape[0] - 3

        next_observation = self._get_obs()
        reward = self.portfolio[self.ticker1] * (next_observation[self.ticker1]  - observation [self.ticker1]) + self.portfolio[self.ticker2] * (next_observation[self.ticker2] - observation[self.ticker2]) + fees
        
        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, info
