# version: 23.03.2020

import numpy as np


# LinearRegression class provides means to apply a linear regression algorithm 
# to a feature matrix X in R^(m x n), label vector y in R^m with
# number of training examples m, number of features n
# X and y are numpy arrays
# using standardization typically improves convergence of gradient descent
class LinearRegression:
    
    def __init__ (self, X, y, standardization=True, addPolyFeats=False, degree=2):
        
        # add polynomial features in bivariate case
        if addPolyFeats == True:
            X = self.addPolynomialFeatures(X, degree)
        
        # multivariate case
        try:
            self.m, self.n = X.shape
        
        # univariate case: reshape feature matrix X
        except:
            X = np.array([X]).T
            self.m, self.n = X.shape
        
        self.X = X
        self.y = y
        self.standardization = standardization
        
        self.mu    = self.X.mean(axis=0)
        self.sigma = self.X.std(axis=0)
        self.theta = np.zeros(self.n + 1)
        
        if standardization == True:
            self.Xnorm = self.standardizeFeatures()
            self.Xfeat = self.addOffset(self.Xnorm)
        
        else:
            self.Xfeat = self.addOffset(self.X)
    
    
    def standardizeFeatures (self):
        return ((self.X - self.mu) / self.sigma)
    
    
    def addOffset (self, X):
        return (np.concatenate((np.ones((self.m, 1)), X), axis=1))
    
    
    # works only for bivariate feature matrix X
    def addPolynomialFeatures (self, X, degree=2):
        
        if X.ndim == 1:
            X = np.array([X]).T
        
        if X.shape[1] != 2:
            raise ValueError("Feature matrix X must contain exactly two columns.")
        
        X1 = np.array([X[:, 0]]).T
        X2 = np.array([X[:, 1]]).T
        
        for i in range(1, degree + 1):
            for j in range(i + 1):
                if i > 1:
                    X = np.concatenate((X, np.power(X1, i-j) * np.power(X2, j)), 
                                       axis=1)
        
        return (X)
    
    
    def cost (self, regularization=0.0):
        
        Xtheta_y = self.Xfeat @ self.theta - self.y
        regTerm  = regularization * self.theta[1:] @ self.theta[1:]
        J        = (Xtheta_y.T @ Xtheta_y + regTerm) / (2.0 * self.m)
        
        return (J)
    
    
    def gradient (self, regularization=0.0):
        
        dJ      = self.Xfeat.T @ (self.Xfeat @ self.theta - self.y)
        dJ[1:] += regularization * self.theta[1:]
        dJ      = dJ / self.m
        
        return (dJ)
    
    
    def gradientDescent (self, iterations=100, learningRate=1e-3, 
                         regularization=0.0):
        
        self.theta = np.zeros(self.n + 1)
        J_history  = np.zeros(iterations)
        
        for i in range(iterations):
            
            self.theta   = self.theta - learningRate * self.gradient(regularization)
            J_history[i] = self.cost(regularization)
            
            # progress indicator
            if (i % (iterations / 100.0) == 0.0):
                print(i * 100.0 / iterations, " %")
        
        print("J = ", J_history[-1])
        
        return (J_history)
    
    
    def normalEquation (self, regularization=0.0):
        
        if regularization == 0.0:
            self.theta = np.linalg.pinv(self.Xfeat.T @ self.Xfeat) \
                         @ self.Xfeat.T @ self.y
        
        else:
            
            regTerm       = regularization * np.identity(self.n + 1)
            regTerm[0, 0] = 0
            
            self.theta = np.linalg.pinv(self.Xfeat.T @ self.Xfeat + regTerm) \
                         @ self.Xfeat.T @ self.y
        
        return (self.theta)
    
    
    def predict (self, X):
        
        if self.standardization == True:
            X = (X - self.mu) / self.sigma
        
        if type(X) == float:
            X = np.array([X])
        
        elif type(X) == list:
            X = np.array(X)
        
        if self.theta.size == 2 and X.ndim < 2:
            X = np.array([X]).T
        
        elif self.theta.size > 2 and X.ndim < 2:
            X = np.array([X])
            
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        return (X @ self.theta)
    
    
    def getCost (self, theta0List, theta1List, regularization=0.0):
        
        if self.n == 1:
            
            theta_prev = self.theta
            
            # creates mesh of theta values
            Theta0, Theta1 = np.meshgrid(theta0List, theta1List)
            
            Cost = np.zeros(Theta0.shape)
            i    = 0
            j    = 0
            
            for i in range(Theta0.shape[0]):
                
                for j in range(Theta0.shape[1]):
                    self.theta = np.array([Theta0[i, j], Theta1[i, j]])
                    Cost[i, j] = self.cost(regularization)
                    j += 1
                
                i += 1
            
            self.theta = theta_prev
            
            return (Theta0, Theta1, Cost)
            
        elif self.n >= 2:
            print("weight vector dimension is larger than 2")
            pass



# LogisticRegression class provides means to apply a logistic regression 
# algorithm to a feature matrix X in R^(m x n), label vector y in R^m with
# number of training examples m, number of features n
# X and y are numpy arrays
# using standardization typically improves convergence of gradient descent
class LogisticRegression:
    
    def __init__ (self, X, y, standardization=True, addPolyFeats=False, degree=2):
        
        # add polynomial features in bivariate case
        if addPolyFeats == True:
            X = self.addPolynomialFeatures(X, degree)
        
        # multivariate case
        try:
            self.m, self.n = X.shape
        
        # univariate case: reshape feature matrix X
        except:
            X = np.array([X]).T
            self.m, self.n = X.shape
        
        self.X = X
        self.y = y
        self.standardization = standardization
        
        self.mu    = self.X.mean(axis=0)
        self.sigma = self.X.std(axis=0)
        self.theta = np.zeros(self.n + 1)
        
        if standardization == True:
            self.Xnorm = self.standardizeFeatures()
            self.Xfeat = self.addOffset(self.Xnorm)
        
        else:
            self.Xfeat = self.addOffset(self.X)
    
    
    def standardizeFeatures (self):
        return ((self.X - self.mu) / self.sigma)
    
    
    def addOffset (self, X):
        return (np.concatenate((np.ones((self.m, 1)), X), axis=1))
    
    
    # works only for bivariate feature matrix X
    def addPolynomialFeatures (self, X, degree=2):
        
        if X.ndim == 1:
            X = np.array([X]).T
        
        if X.shape[1] != 2:
            raise ValueError("Feature matrix X must contain exactly two columns.")
        
        X1 = np.array([X[:, 0]]).T
        X2 = np.array([X[:, 1]]).T
        
        for i in range(1, degree + 1):
            for j in range(i + 1):
                if i > 1:
                    X = np.concatenate((X, np.power(X1, i-j) * np.power(X2, j)), 
                                       axis=1)
        
        return (X)
    
    
    def sigmoid (self, x):
        return (1.0 / (1.0 + np.exp(-x)))
    
    
    def cost (self, regularization=0.0):
        
        h       = self.sigmoid(self.Xfeat @ self.theta)
        regTerm = regularization * self.theta[1:] @ self.theta[1:]
        J       = - (self.y @ np.log(h) + (1 - self.y) @ np.log(1 - h)) / self.m \
                  + regTerm / (2.0 * self.m)
        
        return (J)
    

    def gradient (self, regularization=0.0):
        
        dJ      = self.Xfeat.T @ (self.sigmoid(self.Xfeat @ self.theta) - self.y)
        dJ[1:] += regularization * self.theta[1:]
        dJ      = dJ / self.m
        
        return (dJ)
    
    
    def gradientDescent (self, iterations=100, learningRate=1e-3, 
                         regularization=0.0):
        
        self.theta = np.zeros(self.n + 1)
        J_history  = np.zeros(iterations)
        
        for i in range(iterations):
            
            self.theta   = self.theta - learningRate * self.gradient(regularization)
            J_history[i] = self.cost(regularization)
            
            # progress indicator
            if (i % (iterations / 100.0) == 0.0):
                print(i * 100.0 / iterations, " %")
        
        print("J = ", J_history[-1])
        
        return (J_history)
    
    
    def normalEquation (self, regularization=0.0):
        
        if regularization == 0.0:
            self.theta = np.linalg.pinv(self.Xfeat.T @ self.Xfeat) \
                         @ self.Xfeat.T @ self.y
        
        else:
            
            regTerm       = regularization * np.identity(self.n + 1)
            regTerm[0, 0] = 0
            
            self.theta = np.linalg.pinv(self.Xfeat.T @ self.Xfeat + regTerm) \
                         @ self.Xfeat.T @ self.y
        
        return (self.theta)
    
    
    def predict (self, X):
        
        if self.standardization == True:
            X = (X - self.mu) / self.sigma
        
        if type(X) == float:
            X = np.array([X])
        
        elif type(X) == list:
            X = np.array(X)
        
        if self.theta.size == 2 and X.ndim < 2:
            X = np.array([X]).T
        
        elif self.theta.size > 2 and X.ndim < 2:
            X = np.array([X])
        
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        return (self.sigmoid(X @ self.theta))
