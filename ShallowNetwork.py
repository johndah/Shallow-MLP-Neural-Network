'''
Created on 2 mars. 2018

@author: John Henry Dahlberg
'''

from numpy import *
import time
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import *
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib.offsetbox import AnchoredText

#   cd  C:\Users\johnd\OneDrive\Documents\DD2437 Artificial Neural Networks and Deep Architectures\Lab 1\Part 2
#   python MLP.py

class multiLayerPerceptron(object):
    def __init__(self, X, targets, setProportions, hiddenLayerSizes, eta, alpha, lmbda, tol, sigmaNoise, nEpochs, printProccess, plotType):

        self.N = shape(X)[1]
        self.X = X
        self.T = targets

        self.trainPart = setProportions[0]/(sum(setProportions))
        self.validationPart = setProportions[1]/(sum(setProportions))
        self.testPart = setProportions[2]/(sum(setProportions))

        self.Ntrain = int(self.N*self.trainPart)
        self.Xtrain =  X[:, 0:self.Ntrain]
        self.Ttrain = targets[:, 0:self.Ntrain]

        self.Nvalid = int(self.N*self.validationPart)
        self.Xvalid =  X[:, self.Ntrain:self.Ntrain+self.Nvalid]
        self.Tvalid = targets[:, self.Ntrain:self.Ntrain+self.Nvalid]

        self.Ntest = self.N - (self.Ntrain+self.Nvalid)
        self.Xtest =  X[:, self.Ntrain+self.Nvalid:-1]
        self.Ttest = targets[:, self.Ntrain+self.Nvalid:-1]

        self.normalizeData()

        self.inputLayerSize = shape(X)[0]
        self.outputLayerSize = shape(targets)[0]
        self.hiddenLayerSizes = hiddenLayerSizes

        self.eta = eta
        self.alpha = alpha
        self.lmbda = lmbda
        self.tol = tol
        self.sigmaNoise = sigmaNoise
        self.nEpochs = nEpochs
        self.printProccess = printProccess
        self.plotType = plotType

        self.W = []
        self.deltaW = []
        layerNeuronSizes = copy.copy(hiddenLayerSizes)
        layerNeuronSizes.insert(0, self.inputLayerSize)
        layerNeuronSizes.append(self.outputLayerSize)
        self.layerNeuronSizes = layerNeuronSizes

        for i in range(0, len(layerNeuronSizes)-1):
            random.seed(i)
            self.W.append(0.01*random.randn(layerNeuronSizes[i+1], layerNeuronSizes[i] + 1))
            self.deltaW.append(zeros([layerNeuronSizes[i+1], layerNeuronSizes[i] + 1]))


    def multiLayerPerceptronAlgorithm(self):
        trainRmsErrors = []
        validRmsErrors = []

        nIterations = self.nEpochs
        validRmsErrorPrevious = -1
        earlyStopped = False
        earlyStoppingIteration = nIterations-1

        if self.plotType == 'points':
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(8, 8))

        for iteration in range(0, nIterations):

            # Train neural network
            output, activationsBiased = self.forward(self.Xtrain)
            self.backpropagation(output, activationsBiased)
            trainRmsError = 0.5*sqrt(sum(square(output - self.Ttrain)))/self.Ntrain
            trainRmsErrors.append(trainRmsError)

            # Apply neural network on validation data each iteration
            outputValid, activationsBiasedValid = self.forward(self.Xvalid)
            validRmsError = 0.5*sqrt(sum(square(outputValid - self.Tvalid)))/self.Nvalid
            validRmsErrors.append(validRmsError)

            if self.plotType == 'points' and iteration%(self.nEpochs/200) == 0: # 200
                plt.cla()
                self.plotDataPoints(self.Xtest, self.Ttest, ['b', 'g'])
                self.plotLevelCurve(ax)
            elif self.plotType == 'timeSeries' and iteration%(self.nEpochs/200) == 0:
                plt.cla()
                self.plotTestPerformance(output, self.Ttrain, 'Training')
                plt.pause(0.001)


            if abs(validRmsError - validRmsErrorPrevious) < self.tol and iteration > 0.6*self.nEpochs and not earlyStopped:
                self.earlyStoppingW = self.W
                earlyStoppingIteration = iteration
                nIterations = earlyStoppingIteration + min(int(0.1*(earlyStoppingIteration)), (self.nEpochs-earlyStoppingIteration))
                earlyStopped = True
            elif not earlyStopped:
                    self.earlyStoppingW = self.W

            validRmsErrorPrevious = copy.copy(validRmsError)

            if self.printProccess and iteration%(self.nEpochs/1000) == 0:
                print('Iteration process:', str(100*iteration/self.nEpochs)+' %')

        if earlyStopped:
            stopMessage = 'Stopped due to validation error \n    changing less than tolerance '+ str(self.tol)
            print(stopMessage)
        else:
            stopMessage = 'Stopped due to reaching max number of epochs '
            print(stopMessage)

        self.W = self.earlyStoppingW
        outputTest, activationsBiasedTest = self.forward(self.Xtest)

        if self.plotType == 'timeSeries':
            plt.figure()
            self.plotTestPerformance(outputTest, self.Ttest, 'Test')

        self.plotLearningCurve(trainRmsErrors, validRmsErrors, earlyStoppingIteration, stopMessage)


        return validRmsErrors[earlyStoppingIteration]

    def normalizeData(self):
        scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))

        scaler.fit(self.Xtrain.T)
        self.Xtrain = scaler.transform(self.Xtrain.T).T
        self.Xvalid = scaler.transform(self.Xvalid.T).T
        self.Xtest = scaler.transform(self.Xtest.T).T

        scaler.fit(self.Ttrain.T)
        self.Ttrain = scaler.transform(self.Ttrain.T).T
        self.Tvalid = scaler.transform(self.Tvalid.T).T
        self.Ttest = scaler.transform(self.Ttest.T).T

    def sigmoid(self, Z):
        return 2/(1 + exp(-Z)) - 1

    def forward(self, X):
        N = shape(X)[1]
        activationsBiased = []

        biasTerms = ones((1, shape(X)[1]))
        activationsBiased.append(vstack([X, biasTerms]))

        for i in range(0, len(self.layerNeuronSizes)-1):
            weightedSum = dot(self.W[i], activationsBiased[i])
            activation = self.sigmoid(weightedSum)
            activationsBiased.append(vstack([activation, biasTerms]))

        output = activation
        return output, activationsBiased


    def backpropagation(self, output, activationsBiased):
        RmsErrors = []
        deltaW = self.deltaW
        W = self.W

        eta = self.eta
        alpha = self.alpha
        XBiased = vstack([self.Xtrain, ones((1, self.Ntrain))])

        deltas = []
        finalLayer = len(self.layerNeuronSizes) - 1

        for layer in range(finalLayer, 0, -1):
            if layer == finalLayer:
                dEdA = (1 + output)*(1 - output)/2
                dAdW = (output - self.Ttrain)
                deltas.append(dEdA*dAdW)
            else:
                dEdA = (1 + activationsBiased[layer])*(1 - activationsBiased[layer])/2
                dAdW = dot(self.W[layer].T, deltas[finalLayer - (layer+1)])
                deltas.append(dEdA*dAdW)
                deltas[finalLayer - layer] = deltas[finalLayer - layer][0:self.hiddenLayerSizes[layer-1], :]

            regularizationW = self.lmbda*sum(self.W[layer-1])/self.Ntrain

            deltaW[layer-1] = alpha*deltaW[layer-1] - (1-alpha)*(dot(deltas[finalLayer - layer], activationsBiased[layer-1].T) - regularizationW)
            W[layer-1] += eta*deltaW[layer-1]

        self.deltaW = deltaW
        self.W = W

    def plotLearningCurve(self, trainRmsErrors, validRmsErrors, earlyStoppingIteration, stopMessage):
        nIterations = self.nEpochs
        mpl.rcParams.update(mpl.rcParamsDefault)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)

        constants = 'Max Epochs = ' + str(self.nEpochs) + '\n' + r'$\eta$ = ' + str(self.eta) + \
        '\n# Hidden layers = ' + str(len(self.hiddenLayerSizes)) \
        + '\n# Hidden neurons = ' + str(self.hiddenLayerSizes) \
        + '\n'  + r'$\alpha$ = ' + str(self.alpha) + '\n' + r'$\lambda$ = ' + str(self.lmbda)  \
        + '\n' + 'Tolerance = ' + str(self.tol) + '\n' \
        + r'$\sigma_{Noise}$ = ' + str(self.sigmaNoise) + '\n' \
        + 'Training-, validation- & test proportion \n    = ['\
        + str(round(self.trainPart, 2)*100) + '%, ' + str(round(self.validationPart, 2)*100) + '%, ' + str(round(self.testPart, 2)*100) + '%] \n\n' + stopMessage
        anchored_text = AnchoredText(constants, loc=5)
        ax.add_artist(anchored_text)

        # Errors
        iterations = list(range(0, nIterations))
        iterations = iterations[:nIterations]
        earlyStoppingError = validRmsErrors[earlyStoppingIteration]
        plt.plot(iterations, trainRmsErrors[:nIterations], linewidth=3, label = 'Training')
        plt.plot(iterations, validRmsErrors[:nIterations], linewidth=3, label = 'Validation')
        plt.plot(earlyStoppingIteration+1, earlyStoppingError, 'go', markersize=7)
        plt.text(earlyStoppingIteration*0.8, earlyStoppingError*1.1, 'Early stopping: (' + str(earlyStoppingIteration+1)+', ' + str(round(earlyStoppingError, 5)) + ')', fontsize=12)
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Root mean square error ')
        plt.title( 'Learning curve of MLP')
        plt.grid()

    def plotDataPoints(self, X, targets, colors):
        mT = min(targets[0])
        for i in range(0, len(colors)):
            target = (1 - mT)*i+mT
            classIndices = where(targets == target)[1]
            x = X[0, classIndices]
            y = X[1, classIndices]
            plt.plot(x, y, colors[i]+'o', label = 'Class'+str(target))

        plt.legend(loc='upper right')


    def plotLevelCurve(self, ax):
        x = arange(-2, 2, 0.1)
        y = arange(-2, 2, 0.1)
        xx, yy = meshgrid(x, y)
        nPoints = xx.shape[0]*xx.shape[1]
        xCoords = xx.reshape((nPoints, 1))
        yCoords = yy.reshape((nPoints, 1))
        input = concatenate((xCoords, yCoords), axis=1).T

        W1 = self.W[0]

        for i in range(W1.shape[0]):
            W1i = W1[i]
            x1 = array([-2, 2])
            x2 = -(W1i[0]*x1 + W1i[-1])/W1i[1]
            if i < W1.shape[0] - 1:
                plt.plot(x1, x2, '--', color='orange', linewidth=2, alpha = 0.5)
            else:
                plt.plot(x1, x2, '--', color='orange', linewidth=2, alpha = 0.5, label = 'Normal line to first layer weights')
        plt.plot([], [], 'k', label = 'Level curve of sigmoidal output')
        plt.legend(loc='upper right')


        output, activationsBiased = self.forward(input)
        output = output.reshape(xx.shape)

        ax.collections = [] # Erase the old contour
        ax.contour(x, y, output, (0.5,), color='black')
        plt.title('MLP, ' + str(self.hiddenLayerSizes) + ' hidden neurons')
        plt.axis('equal')
        plt.axis([-2, 2, -2, 2])
        plt.pause(.001)


    def plotTestPerformance(self, output, targets, dataSet):
        N = targets.shape[1]
        iterations = list(range(0, N))
        plt.plot(iterations, targets[0], color=[0,.7, 0], linewidth=3.4, label = dataSet + ' target')
        plt.plot(iterations, output[0], linewidth=2.3, label = dataSet + ' predictor')
        plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.ylabel('X(t)')
        plt.title('Performance on ' + dataSet + ' data with ' +  r'$\sigma_{Noise}$ = ' + str(self.sigmaNoise) + ' and ' + str(self.hiddenLayerSizes) + ' hidden neurons')


def generateCircleData(numPoints):
    random.seed(4)
    XLinNonSep, y = make_circles(n_samples=numPoints, factor=.3, noise=.05)
    XLinNonSep = 10 * XLinNonSep.T
    targets = array([y])

    return XLinNonSep, targets


def generateChaoticTimeSeries(sigmaNoise):

    t0 = 301
    tEnd = 1500
    T = arange(t0, tEnd+1)

    x = zeros((1, tEnd+6))
    x = x[0]

    x[0] = 1.5
    for t in range(1, tEnd+6):
        if t < 25:
            x[t] = x[t - 1] - 0.1*x[t-1]
        else:
            x[t] = x[t - 1] + 0.2*x[t-26]/(1 + x[t-26]**10) - 0.1*x[t-1]

    X = zeros(shape(T))
    for tau in range(20, -5, -5):
        x2 = x[T-tau]
        X = array(vstack([X, x2]))

    targets = x[T+5]

    X = X[1:len(X),:]
    X = array(X)
    #targets = array(targets)
    targets = array([targets])

    random.seed(0)
    XNoised = X + array(sigmaNoise*random.randn(shape(X)[0], shape(X)[1]))
    targetsNoised = targets + array(sigmaNoise*random.randn(shape(targets)[0], shape(targets)[1]))

    return XNoised, targetsNoised

def classification():
    ''' Classify linearly non seperable data '''
    nEpochs = 2500  # Maximum number of ephocs
    eta = 0.004  # Learning rate
    alpha = 0.9  # Momentum
    lmbda = 2e-1  # Regularization
    tol = 2e-6  # Tolerance of validation error change
    hiddenLayerSizes = [3]  # Number of neurons in each optional hidden layer (Dynamic vector with arbitrary size)
    setProportions = [6, 2, 2]  # Proportion sets of data for train, validation and test part (Fixed to the tree sets)
    printProccess = True
    NCirclePoints = 600
    sigmaNoise = 0
    plotType = 'points'
    XLinNonSep, targetsLinNonSep = generateCircleData(NCirclePoints)

    # Initialize MLP with class attributes
    mlp = multiLayerPerceptron(XLinNonSep, targetsLinNonSep, setProportions, hiddenLayerSizes, eta, alpha, lmbda, tol,
                               sigmaNoise, nEpochs, printProccess, plotType)
    # Perform the the training algorithm and show learning curve and final test
    rmsValid = mlp.multiLayerPerceptronAlgorithm()

def prediction():
    ''' Predict chaotic time series '''
    sigmaNoise = 0
    X, targets = generateChaoticTimeSeries(sigmaNoise)

    # Initialize parameters
    nEpochs = 4000  # Maximum number of ephocs
    eta = 0.001  # Learning rate
    alpha = 0.9  # Momentum
    lmbda = 2e-1  # Regularization
    tol = 5e-7  # Tolerance of validation error change
    hiddenLayerSizes = [6, 6]  # Number of neurons in each optional hidden layer (Dynamic vector with arbitrary size)
    setProportions = [6, 2, 2]  # Proportion sets of data for train, validation and test part (Fixed to the tree sets)
    printProccess = True
    plotType = 'timeSeries'

    # Initialize MLP with class attributes
    mlp = multiLayerPerceptron(X, targets, setProportions, hiddenLayerSizes, eta, alpha, lmbda, tol, sigmaNoise, nEpochs, printProccess, plotType)
    # Perform the the training algorithm and show learning curve and final test
    rmsValid = mlp.multiLayerPerceptronAlgorithm()


def main():

    classification()
    prediction()


if __name__ == '__main__':
    main()
    plt.show()
