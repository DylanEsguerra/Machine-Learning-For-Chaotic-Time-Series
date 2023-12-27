#Class to simulate data from discrete Dynamic Systems 


import numpy as np

class DataSimulator:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.dynamics = None
        self.values = None

    def simulate(self):
        raise NotImplementedError("Subclasses must implement the simulate method")

class LogisticMapSimulator(DataSimulator):
    def __init__(self, r, obs, timesteps):
        super().__init__(timesteps)
        self.r = r
        self.obs = obs

    def simulate(self):
        np.random.seed(0)
        time = np.arange(self.timesteps)
        self.dynamics = np.random.rand(self.timesteps, 1)

        for t in range(self.timesteps-1):
            self.dynamics[t+1] = self.r * self.dynamics[t] * (1 - self.dynamics[t])

        self.values = self.dynamics + self.obs * np.std(self.dynamics) * np.random.rand(len(self.dynamics), 1)
        self.values = self.values.reshape(-1, 1)

        # Apply burn-in and discard the first 100 steps
        burn_in = 100
        self.dynamics = self.dynamics[burn_in:]
        self.values = self.values[burn_in:]

        return self.values

