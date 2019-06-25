import pytest


reg = TargetResamplingRegressor(regressor=LinearRegression(),
                                resampler=SimpleTargetResampler(step_size=10))
X = np.ones((95, 4))
y = np.ones(100000)
