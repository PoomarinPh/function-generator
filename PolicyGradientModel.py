import numpy as np
from RewardCalculator import RewardCalculator

class PolicyGradientModel:
    def __init__(self,
                 model,
                 allowedSymbol,
                 numSymbol,
                 maxLength,
                 rewardCalculator,
                 learningRate,
                 fileName):
        '''
        Initialize PolicyGradientModel.

        :param model (keras.modesl.Model): A model to be trained. Assume it's recurrence so the input doesn't require
            the previous input as an input
        :param allowedSymbol (List<String>): List of available (case sensitive) variables e.g. ['X','Y']
        :param maxLength (Int): Max length of the output function
        :param rewardCalculator (RewardCalculator): Calculator for the reward
        :param learningRate (Float): Learning rate for this model
        :param fileName (String): Relative/Absolute path and file name to which the weight of the model will be saved.
        '''
        self.model = model
        self.allowedSymbol = np.array(allowedSymbol)
        self.numSymbol = len(self.allowedSymbol)
        self.maxLength = maxLength
        self.rewardCalculator = rewardCalculator
        self.learningRate = learningRate
        self.fileName = fileName

        self.outputProbsHistory = []
        self.outputSequenceHistory = []
        self.gradients = []
        self.rewards = [] # in case of sequential rewards

    def getModel(self):
        return self.model

    def train(self, input, numIterationPerEpoch=10, numEpoch=10000, numEpochToSaveWeight=10):
        '''
        The procedure of this training is:
            1. Predict output sequence (e.g. "3*X+5") from current model and weight. Store output and probability from
            the model in every step that the model predict.
            2. Calculate reward from the given output sequence (e.g. "3*X+5" -> reward: 15). If the reward > 0 then the
            outcome is good. Backpropagate all predicted output to make model predict more like this output and vice
            versa.
            3. Repeat step 1. This count as 1 epoch
        :param input: Input for the model
        :param numEpoch: Number of epoch to train
        :param numEpochToSaveWeight: Number of epoch that model will be periodically saved.
        '''

        for ep in range(numEpoch):
            averageLoss = 0.0
            for it in range(numIterationPerEpoch):
                # Predict an output sequence
                modelOutput, probHistory, rewardHistory = self.predictOutputSequence(input)
                outputLength = len(modelOutput)

                # Save state of current sequence
                self.__saveState(outputs=modelOutput, probs=probHistory, rewards=rewardHistory)

                if self.allowedSymbol[modelOutput[len(modelOutput)-1]] == '#':
                    outputAlphabet = self.allowedSymbol[modelOutput[0:(len(modelOutput)-1)]]
                else:
                    outputAlphabet = self.allowedSymbol[modelOutput]

                # TODO: Use rewards of each character in the sequence to do backpropagation
                reward = self.rewardCalculator.calReward(''.join(outputAlphabet))
                gradients = np.vstack(self.gradients)
                gradients *= reward
                X = np.ones((outputLength,1,1))
                reshapedOutputProbsHistory = np.array(self.outputProbsHistory).reshape(outputLength, self.numSymbol)
                Y = reshapedOutputProbsHistory + self.learningRate * np.squeeze(np.vstack([gradients]))
                self.model.reset_states()
                loss = self.model.train_on_batch(X, Y)
                averageLoss += loss
                self.__resetHistory()

            averageLoss /= numIterationPerEpoch
            print("Epoch: %d\tLoss: %s\tExample Output: %s" %(ep, averageLoss, ''.join(outputAlphabet)))

            if ep % numEpochToSaveWeight == 0:
                print("Saving Weight")
                self.saveWeight()


    def predictOutputSequence(self, input):
        """
        Predict output sequence from the given input.
        :param input:
        :return (List<Int>,List<List<float>>): List of index of characters in the sequence Dim: length of sequence.
        List of list of probability of each character of each character in the sequence Dim: length of sequence and
        numSymbol.
        """
        outputs = []
        outputProbHistory = []
        rewards = []

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
            outputs.append(output)

            # TODO: Calculate reward from current sequence and append on rewards

            if self.allowedSymbol[output] == '#':
                break
        return outputs, outputProbHistory, rewards

    def saveWeight(self):
        '''
        Save weight of the model from the path specified at init.
        '''
        self.model.save(self.fileName)

    def loadWeight(self):
        '''
        Load weight of the model from the path specified at init
        '''
        self.model.load_weights(self.fileName)

    def __resetHistory(self):
        self.outputSequenceHistory = []
        self.outputProbsHistory = []
        self.gradients = []
        self.rewards = []

    def __saveState(self, outputs, probs, rewards):
        for i in range(len(probs)):
            self.outputProbsHistory.append(probs[i])
            self.outputSequenceHistory.append(outputs[i])
            self.rewards.append(rewards[i])

            y = np.zeros([self.numSymbol])
            y[outputs[i]] = 1
            self.gradients.append(y.astype('float32') - probs[i])