#from Model_Output_Function import outputFunction
import numpy as np
import os

class Cost_Calculator:
    def __init__(self, parameters, correctExpression, usingFile=False):
        self.outputDifferenceWeight = 1 # Multiply to the difference
        self.outputCompilableWeight = +150 #
        self.differenceCap = np.exp(100) # More than or less than negative of this will be capped
        self.outputLengthWeight = -3 # per characters
        self.parameters = parameters # List of Variable name
        self.correctExpression = correctExpression # In case the output is expression
        self.usingFile = usingFile

    def calReward(self, expression):
        compilableReward, differenceReward  = self.__calCompileAndDifferenceReward(expression)
        lengthReward = self.__calLengthReward(expression)
        print(differenceReward, compilableReward, lengthReward)
        return differenceReward + compilableReward + lengthReward

    def __calCompileAndDifferenceReward(self, expression):
        # In case of using output file
        if self.usingFile:
            self.__editOutputFile(expression)

        sumDiff = 0
        for x in range(30):
            for y in range(30):
                try:
                    if self.usingFile:
                            sumDiff += self.__calDifferenceWithFile(x,y)
                    else:
                        sumDiff += self.__calDifferenceWithEval(expression,x,y)
                except SyntaxError:
                    return -self.outputCompilableWeight, 0

        if sumDiff > self.differenceCap:
            sumDiff = self.differenceCap
        if sumDiff == 0:
            sumDiff = np.exp(-100)

        return self.outputCompilableWeight, self.__scaleValue(sumDiff)

    def __calLengthReward(self, expression):
        return len(expression) * self.outputLengthWeight

    def __scaleValue(self, value):
        return - np.log(value)

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
        ebsilon = 10 ** -100

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

        #print(X,Y,correctVal,outputVal)

        return abs(correctVal - outputVal)

# costCalculator = Cost_Calculator(parameters=['X','Y'], correctExpression="3*X+2*Y")
# for i in range(0,10):
#     reward = costCalculator.calReward("3*X+2*Y")
# print(reward)