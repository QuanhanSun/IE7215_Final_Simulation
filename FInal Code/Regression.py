import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats


class regression():
	def __init__(self, FerData, Time):
		self.FerData = FerData
		self.Time = Time
		self.grate = 0
		self._P = 0
		self.sig_P = 0
		self.alpha = 0
		self._I = 0
		self.sig_I = 0

	def bootstrap(self):
		index = np.random.choice(len(self.FerData.iloc[:, 2]), len(self.FerData.iloc[:, 2]))
		self.FerData = self.FerData.iloc[index, :]

	def reg(self):
		y = np.log(self.FerData.iloc[:, 2]) - np.log(self.FerData.iloc[:, 1])
		y = y.values.reshape(-1, 1)
		X = np.array([self.Time for i in range(len(y))])
		X = X.reshape(-1, 1)
		clf = LinearRegression(fit_intercept=False)
		clf.fit(X, y)
		y_hat = clf.predict(X)
		self.grate = round(clf.coef_[0][0], 4)
		self.P = y - y_hat
		self.sig_P = round(np.std(self.P, ddof=1), 4)
		y2 = self.FerData.iloc[:, 3] / self.FerData.iloc[:, 2]
		y2 = y2.values.reshape(-1, 1)
		self.alpha = round(pow(np.prod(y2), 1.0 / len(y2)), 4)
		y_hat2 = self.FerData.iloc[:, 2] * self.alpha
		y_hat2 = y_hat2.values.reshape(-1, 1)
		exp_I = self.FerData.iloc[:, 3].values.reshape(-1, 1) / y_hat2
		self._I = np.log(exp_I)
		self.sig_I = round(np.std(self._I, ddof=1), 4)


