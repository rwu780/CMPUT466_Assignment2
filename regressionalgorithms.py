from __future__ import division  # floating point division
import numpy as np
import math

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

    # Error functions copy from script_regression.py
    def geterror(self, predictions, ytest):
        # Can change this to other error values
        return 0.5*self.l2err(predictions,ytest)**2/len(ytest)

    def l2err(self, prediction,ytest):
        """ l2 error (i.e., root-mean-squared-error) """
        return np.linalg.norm(np.subtract(prediction,ytest))


class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.5}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain, lamda = 0.01):
        # Overwrite the super class method
        # update the weight with l2 regularizer
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain) + (lamda*np.eye(Xtrain.shape[1]))), Xtrain.T), ytrain)

class StochasticGradientDescent(Regressor):
    """
    Stochastic Gradient Descent 
    Implement according to pseudo code in notes

    """

    def __init__( self, parameters = {}):
        # Default parameters
        self.params = {'regwgt': 0.5}
        self.reset(parameters)
        
        self.numberofRuns = 0
        self.errMSE = {}
        self.errMSE = np.zeros(1001)

    def learn(self, Xtrain, ytrain, stepsize = 0.01):

        self.numberofRuns += 1

        #initial weights to random numbers
        self.weights = np.ones(Xtrain.shape[1]) * self.params['regwgt']
        self.errMSE[0] = (self.errMSE[0]+self.geterror(np.dot(Xtrain, self.weights), ytrain))/self.numberofRuns

        # Begin learning process
        for i in range(1000):
            for t in range(Xtrain.shape[0]):
                # Calculate the gradient
                predict = np.dot(Xtrain[t].T, self.weights)
                g = (predict - ytrain[t]) * Xtrain[t] / len(ytrain)

                # Update to weight
                self.weights -= 0.01 * g

            self.errMSE[i+1] = (self.errMSE[i+1]+self.geterror(np.dot(Xtrain, self.weights), ytrain))/self.numberofRuns

    def getErrMSE(self):
        return self.errMSE

class BatchGradientDescent(Regressor):
    """
    Batch Gradient Descent
    Implement according to pseudo code in notes

    """

    def __init__(self, parameters = {}):
        # Default parameters
        self.params = {'regwgt': 0.5}
        self.reset(parameters)

        self.length = 0
        self.numberofRuns = 0
        self.errMSE = {}
        self.errMSE = np.zeros(1001)

    def learn(self, Xtrain, ytrain):
        self.numberofRuns += 1;

        # Initial weights to random numbers
        self.weights = np.ones(Xtrain.shape[1]) * self.params['regwgt']
        err = 9999999999999 # Infinity
        tolerance = 10 * math.exp(-4)

        numberOfEpoch = 0
        # Calculate intial Error
        Cost = self.geterror(np.dot(Xtrain, self.weights), ytrain)
        self.errMSE[numberOfEpoch] = (self.errMSE[numberOfEpoch]+Cost)/self.numberofRuns


        # Learning process
        while abs(Cost - err) > tolerance and numberOfEpoch < 1000:
            # Calculate Gradient
            err = Cost
            g = np.dot(Xtrain.T, (np.dot(Xtrain, self.weights) - ytrain))/len(ytrain)

            # Line Search
            stepsize = 1
            alpha = 0.7
            maxIteration = 1000
            iteration = 0
            obj = err
            while iteration < maxIteration:
                w = self.weights - stepsize * g;

                # Ensure improvements is at least as much as tolerance
                e = self.geterror(np.dot(Xtrain, w), ytrain)
                if e < obj - tolerance:
                    break

                # Else the objectives is worse and so decrease stepsize
                stepsize = stepsize * alpha
                iteration += 1
                #obj = e

            # Could not improve solution, stepsize = 0
            if iteration == maxIteration:
                stepsize = 0

            # Weight step toward the min, and update cost function
            self.weights -= stepsize*g
            Cost = self.geterror(np.dot(Xtrain, self.weights), ytrain)
            numberOfEpoch += 1
            
            # Add MSE to list
            self.errMSE[numberOfEpoch] = (self.errMSE[numberOfEpoch]+Cost)/self.numberofRuns

        # Reset the number of epochs
        if(numberOfEpoch >= self.length):
            self.length = numberOfEpoch
        
    def getErrMSE(self):
        return self.errMSE[0:self.length]

class LassoGradientDescent(Regressor):
    """
    Lasso Gradient Descent
    Implement according to Batch Gradient Descent for l1 regularized linear regression
    """
    
    def __init__(self, parameters = {}):
        # Default parameters
        self.params = {'regwgt': 0}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        # Initial weights start at 0
        self.weights = np.zeros(Xtrain.shape[1])

        # Initial Error and Toleerance value
        err = 9999999999999 # Infinity
        tolerance = 10 ** (-5)

        # Start Learning Process
        n = len(ytrain)
        xx = np.dot(Xtrain.T, Xtrain)/n
        xy = np.dot(Xtrain.T, ytrain)/n
        stepsize = 1/(2*np.linalg.norm(xx))

        Err = self.geterror(np.dot(Xtrain, self.weights), ytrain)

        while abs(Err - err) > tolerance:
            err = Err

            # Update the weight according to proximal function
            temp = -1 * stepsize * np.dot(xx, self.weights) + stepsize * xy
            self.weights = self.prox(self.weights + temp, stepsize)

            # Get Error of new weight
            Err = self.geterror(np.dot(Xtrain, self.weights), ytrain)

    def prox(self, w, stepsize, lamda = 0.01):
        """
        Proximal method
        for non-smooth objects
        wi - n * lamda if wi > n * lamda
        0 if abs(wi) <= n * lamda
        wi + n * lamda < -n * lamda
        """

        for i in range(len(w)):
            if(w[i] > lamda * stepsize):
                w[i] -= lamda * stepsize
            elif(abs(w[i]) <= lamda * stepsize):
                w[i] = 0
            elif(w[i] < (-1) * lamda * stepsize):
                w[i] += lamda * stepsize 
        return w


