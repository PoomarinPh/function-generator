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

    def train(self, input, numIterationPerEpoch=100, numEpoch=100000, numEpochToSaveWeight=10):
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
                modelOutput, probHistory = self.predictOutputSequence(input)
                outputLength = len(modelOutput)

                if outputLength <= 1:
                    continue
                
                if self.allowedSymbol[modelOutput[len(modelOutput)-1]] == '#':
                    outputAlphabet = self.allowedSymbol[modelOutput[0:(len(modelOutput)-1)]]
                else:
                    outputAlphabet = self.allowedSymbol[modelOutput]

                # Calculate eventual reward
                reward = self.rewardCalculator.calReward(''.join(outputAlphabet))
                
                # Save state of current sequence
                self.__saveState(outputs=modelOutput, probs=probHistory, isPositiveReward=(reward > 0))
                #print(len(self.gradients))
                #print(len(self.gradients[0]))
                #print(self.gradients)

                gradients = np.vstack(self.gradients)
                gradients *= np.abs(reward)
                X = np.ones((1,outputLength,1))
                reshapedOutputProbsHistory = np.array(self.outputProbsHistory).reshape(outputLength, self.numSymbol)
                Y = reshapedOutputProbsHistory + self.learningRate * np.squeeze(np.vstack([gradients]))
                Y = Y.reshape(1, Y.shape[0], Y.shape[1])
                self.model.reset_states()
                loss = self.model.train_on_batch(X, Y)
                averageLoss += loss
                self.__resetHistory()

            averageLoss /= numIterationPerEpoch
            print("Epoch: %d\tLoss: %s\tExample Output: %s\tExample Reward: " %(ep, averageLoss, ''.join(outputAlphabet)),reward)
            if ep % 10 == 0:
                if len(probHistory)>1:
                    print(probHistory[0])
                    print(probHistory[1])
                elif len(probHistory)==1:
                    print(probHistory[0])
                print(loss)
                print(gradients)
            
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
        self.model.reset_states()
        
        input_batch = np.ones((1, self.maxLength, 1))
        outputProbs = self.model.predict(input_batch, batch_size=self.maxLength)
        #print(outputProbs.shape)
        #print("outputProbs")
        #print(outputProbs.shape)
        
        for i in range(self.maxLength):
            outputProb = outputProbs[0,i,:]
            #print(outputProb.shape)
            normalizedProb = outputProb / np.sum(outputProb)
            #print(normalizedProb)
            outputProbHistory.append(normalizedProb)

            # Random from distribution instead of getting max
            # Due to it's not supervised learning where output is correct/incorrect
            # Output is action. It can be both correct or incorrect
            # Source: http://karpathy.github.io/2016/05/31/rl/
            output = np.random.choice(self.numSymbol, 1, p=normalizedProb.reshape((self.numSymbol)))[0]
            #output = np.argmax(normalizedProb.reshape((self.numSymbol)))
            #print('Output = ', self.allowedSymbol[output])
            outputs.append(output)
            if self.allowedSymbol[output] == '#':
                break
        return outputs, outputProbHistory

    def saveWeight(self):
        '''
        Save weight of the model from the path specified at init.
        '''
        self.model.save(self.fileName)

    def loadWeight(self, filename=None):
        '''
        Load weight of the model from the path specified at init
        '''
        if filename == None:
            self.model.load_weights(self.fileName)
        else:
            self.model.load_weights(filename)

    def __resetHistory(self):
        self.outputSequenceHistory = []
        self.outputProbsHistory = []
        self.gradients = []
        self.rewards = []

    def __saveState(self, outputs, probs, isPositiveReward):
        # TODO(Poomarin): Save necessary variables for back propagation
        for i in range(len(probs)):
            self.outputProbsHistory.append(probs[i])
            self.outputSequenceHistory.append(outputs[i])
            self.rewards.append(0) # Dummy
            if i == len(probs)-1:
                if isPositiveReward:
                    #print("here")
                    #print(probs[i])
                    y = np.zeros([self.numSymbol])
                    y[outputs[i]] = 1
                    self.gradients.append(y.astype('float32') - probs[i])
                else:
                    #print("here2")
                    #print(probs[i])
                    y = np.ones([self.numSymbol])
                    y[outputs[i]] = 0
                    self.gradients.append(y.astype('float32') - probs[i])
            else:
                #print("here3")
                #print(probs[i])
                self.gradients.append(np.zeros([self.numSymbol]))
            #print(self.gradients[len(self.gradients)-1])