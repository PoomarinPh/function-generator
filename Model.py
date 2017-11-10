import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

class PolicyGradientModel:
    def __init__(self,
                 model,
                 allowedParam,
                 allowedSymbol,
                 numSymbol,
                 maxLength):
        self.model = model
        self.allowedParam = allowedParam
        self.allowedSymbol = allowedSymbol
        self.numSymbol = numSymbol
        self.maxLength = maxLength

        self.outputProbHistory = []

    def getModel(self):
        return self.model

    def train(self, epoch=1000):
        for _ in range(epoch):
            modelOutput, probHistory = self.predictOutputSequence()
            self.saveState()
            #TODO(Poomarin): Calculate Reward
            #TODO(Poomarin): Train from reward

    def predictOutputSequence(self, input):
        outputs = []
        outputProbHistory = []
        for i in range(self.maxLength):
            outputProb = self.model.predict(input, batch_size=1)
            outputProbHistory.append(outputProb)
            normalizedProb = outputProb / np.sum(outputProb)

            # Random from distribution instead of getting max
            # Due to it's not supervised learning where output is correct/incorrect
            # Output is action. It can be both correct or incorrect
            # Source: http://karpathy.github.io/2016/05/31/rl/
            output = np.random.choice(self.numSymbol, 1, p=normalizedProb)[0]
            if output != '#':
                outputs.append(output)
            else:
                break
        return outputs, outputProbHistory

    def saveState(self, action, prob):
        # TODO(Poomarin): Save necessary variables for back propagation
        pass

    def saveWeight(self, fileName):
        self.model.save(fileName)

    def loadWeight(self, fileName):
        self.model.load_weights(fileName)