#from Model_Output_Function import outputFunction
import numpy as np
import os
import random

class RewardCalculator:
    def __init__(self, correctExpression, parameters, usingFile=False):
        """
        Initialize reward calculator.
        :param correctExpression (String): Correct expression e.g. "2*X+3*Y"
        :param parameters (List<String>): List of available (case sensitive) variables e.g. ['X','Y']
        :param usingFile (Bool): True to use file-writing to calculate expression result
        """
        self.outputDifferenceWeight = 1 # Multiply to the difference
        self.outputCompilableWeight = +150 #
        self.differenceCap = np.exp(100) # More than or less than negative of this will be capped
        self.outputLengthWeight = -3 # per characters
        self.parameters = parameters # List of Variable name
        self.correctExpression = correctExpression # In case the output is expression
        self.usingFile = usingFile

    def calReward(self, expression):
        """
        Calculate reward.
        :param expression (String): Expression to compare to self.correctExpression
        :return (Double): Reward (Negative = Bad, Positive = Good)
        """
        compilableReward, differenceReward  = self.__calCompileAndDifferenceReward(expression)
        lengthReward = self.__calLengthReward(expression)
        print(differenceReward, compilableReward, lengthReward)
        return differenceReward + compilableReward + lengthReward

    def __calCompileAndDifferenceReward(self, expression):
        # In case of using output file
        if self.usingFile:
            self.__editOutputFile(expression)

        sumDiff = 0.0
        breakOut = False
        for x in range(30):
            for y in range(30):
                try:
                    if self.usingFile:
                        diff = self.__calDifferenceWithFile(x,y)
                        # TODO Add using file
                    else:
                        diff = self.__calDifferenceWithEval(expression,x,y)
                    if self.differenceCap < diff:
                        sumDiff = self.differenceCap
                        breakOut = True
                        break
                    sumDiff += diff
                    if sumDiff > self.differenceCap:
                        sumDiff = self.differenceCap
                        breakout = True
                        break
                except NameError:
                    return -self.outputCompilableWeight, 0
                except SyntaxError:
                    return -self.outputCompilableWeight, 0
                except ZeroDivisionError:
                    return -self.outputCompilableWeight, 0 # Permanent Syntax Error
                except OverflowError:
                    sumDiff = self.differenceCap
                    breakOut = True
                    break
            if breakOut:
                break

        if sumDiff > self.differenceCap:
            sumDiff = self.differenceCap
        if sumDiff == 0:
            sumDiff = np.exp(-100)

        return self.outputCompilableWeight, self.__scaleValue(sumDiff)

    def __calLengthReward(self, expression):
        return len(expression) * self.outputLengthWeight

    def __scaleValue(self, value):
        return -np.log(value)

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
        #outputVal =outputFunction(X,Y)
        #return correctVal - outputVal
        pass

    def __calDifferenceWithEval(self, expression, X, Y):
        ebsilon = 10 ** -30

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


# costCalculator = Cost_Calculator(parameters=['X','Y'], correctExpression="3*X+2*Y")
# letters = '0,1,2,3,4,5,6,7,8,9,X,Y,+,-,*,/'
# letters = letters.split(",")
# for i in range(0,100000000):
#     if i % 10000 == 0:
#         print(i)
#     expression = ""
#     for j in range(0,30):
#         letter_idx = random.randint(0,len(letters)-1)
#         expression += letters[letter_idx]
#     print(expression)
#     reward = costCalculator.calReward(expression)
# print(reward)