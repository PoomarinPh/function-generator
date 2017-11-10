import numpy as np
from RewardCalculator import RewardCalculator

class PolicyGradientModel:
    def __init__(self,
                 model,
                 allowedParam,
                 allowedSymbol,
                 numSymbol,
                 maxLength,
                 correctExpression,
                 learningRate):
        self.model = model
        self.allowedParam = np.array(allowedParam)
        self.allowedSymbol = np.array(allowedSymbol)
        self.numSymbol = numSymbol
        self.maxLength = maxLength
        self.correctExpression = correctExpression
        self.learningRate = learningRate

        self.rewardCalculator = RewardCalculator(correctExpression=correctExpression,
                                                 parameters=allowedParam,
                                                 usingFile=False)

        self.outputProbsHistory = []
        self.outputSequenceHistory = []
        self.gradients = []
        self.rewards = [] # in case of sequential rewards

    def getModel(self):
        return self.model

    def train(self, input, epoch=100000000):
        for _ in range(epoch):

            # Predict an output sequence
            modelOutput, probHistory = self.predictOutputSequence(input)
            outputLength = len(modelOutput)

            # Save state of current sequence
            self.saveState(outputs=modelOutput, probs=probHistory)

            if self.allowedSymbol[modelOutput[len(modelOutput)-1]] == '#':
                outputAlphabet = self.allowedSymbol[modelOutput[0:(len(modelOutput)-1)]]
            else:
                outputAlphabet = self.allowedSymbol[modelOutput]

            reward = self.rewardCalculator.calReward(''.join(outputAlphabet))
            gradients = np.vstack(self.gradients)
            gradients *= reward
            X = np.ones((outputLength,1,1))
            reshapedOutputProbsHistory = np.array(self.outputProbsHistory).reshape(outputLength, self.numSymbol)
            Y = reshapedOutputProbsHistory + self.learningRate * np.squeeze(np.vstack([gradients]))
            self.model.reset_states()
            loss = self.model.train_on_batch(X, Y)

            if epoch % 10000 == 0:
                #print('Epoch is \t'epoch)
                #print('Loss is \t', loss)
                print(''.join(outputAlphabet))


            self.resetHistory()

    def resetHistory(self):
        self.outputSequenceHistory = []
        self.outputProbsHistory = []
        self.gradients = []
        self.rewards = []

    def predictOutputSequence(self, input):
        outputs = []
        outputProbHistory = []
        self.model.reset_states()
        for i in range(self.maxLength):
            outputProb = self.model.predict(input, batch_size=1)
            outputProbHistory.append(outputProb)
            normalizedProb = outputProb / np.sum(outputProb)

            # Random from distribution instead of getting max
            # Due to it's not supervised learning where output is correct/incorrect
            # Output is action. It can be both correct or incorrect
            # Source: http://karpathy.github.io/2016/05/31/rl/
            output = np.random.choice(self.numSymbol, 1, p=normalizedProb.reshape((self.numSymbol)))[0]
            #print('Output = ', self.allowedSymbol[output])
            outputs.append(output)
            if self.allowedSymbol[output] == '#':
                break
        return outputs, outputProbHistory

    def saveState(self, outputs, probs):
        # TODO(Poomarin): Save necessary variables for back propagation
        for i in range(len(probs)):
            self.outputProbsHistory.append(probs[i])
            self.outputSequenceHistory.append(outputs[i])
            self.rewards.append(0) # Dummy

            y = np.zeros([self.numSymbol])
            y[outputs[i]] = 1
            self.gradients.append(y.astype('float32') - probs[i])

    def saveWeight(self, fileName):
        self.model.save(fileName)

    def loadWeight(self, fileName):
        self.model.load_weights(fileName)