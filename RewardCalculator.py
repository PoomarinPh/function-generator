#from Model_Output_Function import outputFunction
import numpy as np
import os
import random

class RewardCalculator:

    def __init__(self,
                 correctExpression,
                 parameters,
                 functionDifferenceRewardWeight=1,
                 compilableRewardWeight=1,
                 lengthRewardWeight=-0.02,
                 rewardOffset=0.0,
                 usingFile=False):
        """
        Initialize reward calculator.
        :param correctExpression (String): Correct expression e.g. "2*X+3*Y"
        :param parameters (List<String>): List of available (case sensitive) variables e.g. ['X','Y']
        :param usingFile (Bool): True to use file-writing to calculate expression result
        """
        self.lnCapEach = 15
        self.capEach = np.exp(self.lnCapEach)
        diffMinCheck = -10 # Included
        diffMaxCheck = 11 # Excluded
        self.diffRangeChecked = range(diffMinCheck, diffMaxCheck)
        self.diffAllCap = np.log((diffMaxCheck - diffMinCheck) * (diffMaxCheck - diffMinCheck) * self.capEach)

        self.outputDifferenceWeight = functionDifferenceRewardWeight # Multiply to the difference
        self.outputCompilableWeight = compilableRewardWeight #
        self.outputLengthWeight = lengthRewardWeight # per characters
        self.parameters = parameters # List of Variable name
        self.correctExpression = correctExpression # In case the output is expression
        self.usingFile = usingFile
        self.rewardOffset = rewardOffset

        #self.maxReward = self.diffAllCap * usingFunctionDifferenceReward + 30 * abs(self.outputLengthWeight) + abs(self.outputCompilableWeight)
        self.maxReward = self.diffAllCap * functionDifferenceRewardWeight + abs(self.outputCompilableWeight)

    def normReward(self, reward):
        if self.maxReward == 0:
            return 0
        return 1.0 * reward / self.maxReward

    def calReward(self, expression):
        """
        Calculate reward.
        :param expression (String): Expression to compare to self.correctExpression
        :return (Double): Reward (Negative = Bad, Positive = Good)
        """
        if expression == self.correctExpression:
            return 1000


        compilableReward, differenceReward  = self.__calCompileAndDifferenceReward(expression)
        lengthReward = self.__calLengthReward(expression)
        if compilableReward < 0:
            return 0

        if compilableReward > 0 and np.abs(differenceReward) < 0.000000001:
            return 10000000
        # reward = self.normReward(differenceReward) + self.rewardOffset
        reward = differenceReward + self.rewardOffset
        reward = reward
        if reward < -1:
            reward = -1
        return reward

    def __calCompileAndDifferenceReward(self, expression):
        # In case of using output file
        if self.usingFile:
            self.__editOutputFile(expression)

        sumDiff = 0.0
        breakOut = False

        rangeVal = np.array(range(-10,11))
        rangeVal = rangeVal ** 2
        rangeVal = rangeVal.tolist()

        for x in rangeVal:
            for y in rangeVal:
                try:
                    if self.usingFile:
                        diff = self.__calDifferenceWithFile(x,y)
                        # TODO Add using file
                    else:
                        diff = self.__calDifferenceWithEval(expression,x,y)
                    if self.capEach < diff:
                        diff = self.capEach
                    sumDiff += diff
                except NameError:
                    return -self.outputCompilableWeight, 0
                except SyntaxError:
                    return -self.outputCompilableWeight, 0
                except ZeroDivisionError:
                    return -self.outputCompilableWeight, 0  # Permanent Syntax Error
                except OverflowError:
                    diff = self.capEach
                    sumDiff += diff
                    breakOut = True
                    break
            if breakOut:
                break

        if sumDiff == 0:
            sumDiff = np.exp(-100)

        return self.outputCompilableWeight, self.__scaleValue(sumDiff) * self.outputDifferenceWeight

    def __calLengthReward(self, expression):
        return len(expression) * self.outputLengthWeight

    def __scaleValue(self, value):
        if value <= 0.000000000001:
            return 0
        scaled = ((1 - np.log(value) / self.diffAllCap) * 2 - 1)
        return scaled

    def __editOutputFile(self, expression):
        try:
            os.remove("Model_Output_Function.py")
        except FileNotFoundError:
            pass
        with open('Model_Output_Function.py', 'w') as file:
            parameters = ",".join(self.parameters)
            file.write("def outputFunction(%s):\n" %(parameters))
            file.write("    print(%s)\n" %(expression))
            file.write("    return " + expression)

    def __calDifferenceWithFile(self, X, Y):
        pass

    def __calDifferenceWithEval(self, expression, X, Y):
        ebsilon = 10 ** -30
        if "**" in expression:
            raise SyntaxError
        try:
            correctVal = eval(self.correctExpression)
        except ZeroDivisionError:
            X += ebsilon
            Y += ebsilon
            correctVal = eval(self.correctExpression)
            X -= ebsilon
            Y -= ebsilon
        try:
            outputVal = eval(expression)
        except ZeroDivisionError:
            X += ebsilon
            Y += ebsilon
            outputVal = eval(expression)
        return abs(correctVal - outputVal)