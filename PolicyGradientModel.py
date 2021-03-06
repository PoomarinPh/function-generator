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
                modelOutput, probHistory, inputSequence = self.predictOutputSequence(np.copy(input))
                outputLength = len(modelOutput)

                # Save state of current sequence
                self.__saveState(outputs=modelOutput, probs=probHistory)

                if self.allowedSymbol[modelOutput[len(modelOutput)-1]] == '#':
                    outputAlphabet = self.allowedSymbol[modelOutput[0:(len(modelOutput)-1)]]
                else:
                    outputAlphabet = self.allowedSymbol[modelOutput]

                reward = self.rewardCalculator.calReward(''.join(outputAlphabet))
                gradients = np.vstack(self.gradients)
                gradients *= reward
                reshapedOutputProbsHistory = np.array(self.outputProbsHistory).reshape(outputLength, self.numSymbol)
                Y = reshapedOutputProbsHistory + self.learningRate * np.squeeze(np.vstack([gradients]))
                Y = Y.reshape(Y.shape[0], Y.shape[1])
                self.model.reset_states()
                loss = self.model.train_on_batch(inputSequence, Y)
                averageLoss += loss
                self.__resetHistory()

            averageLoss /= numIterationPerEpoch
            print("Epoch: %d\tLoss: %s\tExample Output: %s\tExample Reward: " %(ep, averageLoss, ''.join(outputAlphabet)),reward)
            if ep % 50 == 0:
                print("Prob")
                for i in range(len(probHistory)):
                    print(probHistory[i])
                print("Gradient")
                print(gradients)
            
            if ep % numEpochToSaveWeight == 0:
                print("Saving Weight")
                self.saveWeight()

    def __toOneHot(self, idx):
        onehot = np.zeros((self.numSymbol))
        onehot[idx] = 1
        return onehot

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
        inputSequence = []
        self.model.reset_states()
        for i in range(self.maxLength):
            outputProb = self.model.predict(input, batch_size=1)
            outputProbHistory.append(outputProb)
            normalizedProb = outputProb / np.sum(outputProb)

            inputSequence.append(np.copy(input).reshape(input.shape[1], input.shape[2]))

            # Random from distribution instead of getting max
            # Due to it's not supervised learning where output is correct/incorrect
            # Output is action. It can be both correct or incorrect
            # Source: http://karpathy.github.io/2016/05/31/rl/
            output = np.random.choice(self.numSymbol, 1, p=normalizedProb.reshape((self.numSymbol)))[0]
            # print('Output = ', self.allowedSymbol[output])
            onehotOutput = self.__toOneHot(output)
            input[0, i, :] = onehotOutput

            # print(onehotOutput)
            # print(input)



            outputs.append(output)
            if self.allowedSymbol[output] == '#':
                break

                #         for i in range(len(inputSequence)):
                #             print(inputSequence[i])
        inputSequence = np.array(inputSequence)
        #         print(inputSequence.shape)

        return outputs, outputProbHistory, inputSequence

    def saveWeight(self):
        '''
        Save weight of the model from the path specified at init.
        '''
        self.model.save(self.fileName)

    def loadWeight(self, filename=None):
        '''
        Load weight of the model from the path specified at init
        '''
        if filename==None:
            self.model.load_weights(self.fileName)
        else:
            self.model.load_weights(filename)

    def __resetHistory(self):
        self.outputSequenceHistory = []
        self.outputProbsHistory = []
        self.gradients = []
        self.rewards = []

    def __saveState(self, outputs, probs):
        # TODO(Poomarin): Save necessary variables for back propagation
        for i in range(len(probs)):
            self.outputProbsHistory.append(probs[i])
            self.outputSequenceHistory.append(outputs[i])
            self.rewards.append(0) # Dummy

            y = np.zeros([self.numSymbol])
            y[outputs[i]] = 1
            self.gradients.append(y.astype('float32') - probs[i])