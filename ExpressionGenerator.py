import numpy as np
from random import randint

class ExpressionGenerator:
  def __init__(self, model, exp_length=10):
    self.model = model
    self.exp_length = exp_length

    self.characters = "XY0123456789+-*/#"
    self.number = np.array([0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
    self.variable = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    self.end = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    self.symbol = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0])
    

  def __checkType(self, character):
    if character in "XY":
      return 'variable'
    if character in "0123456789":
      return 'number'
    if character in "#":
      return 'end'
    return 'operator'

  def generateExpression(self):

    exp = []

    for i in range(self.exp_length + 3):
      should_end = i > self.exp_length

      if i == 0:
        sample = (self.variable + self.number)
      else:
        type = self.__checkType(char)
        if type == 'variable':
          sample = (self.symbol * 6 + self.end)
          if should_end:
            sample = self.end
        elif type == 'operator':
          sample = (self.variable + self.number)
        elif type == 'number':
          sample = (self.number + self.symbol * 4 + self.end)
          if should_end:
            sample = self.end
        elif type == 'end':
          break

      sample = sample / np.sum(sample)
      pick = np.random.choice(len(self.characters), 1, p=sample)[0]
      char = self.characters[pick]
      exp.append(char)

    return exp

  def to_one_hot(self, exp):
    oh_exp = None
    for e in exp:
      idx = self.characters.index(e)
      one_hot = np.zeros(len(self.characters))
      one_hot[idx] = 1

      if oh_exp is None:
        oh_exp = [one_hot]
      else:
        oh_exp = np.append(oh_exp, [one_hot], axis=0)
    return oh_exp

  def to_character(self, oh_exp):
    oh_exp = np.argmax(oh_exp, axis=1)
    exp = []
    for e in oh_exp:
      exp.append(self.characters[e])
    return exp

if __name__ == '__main__':
  exp = ExpressionGenerator(None)
  for i in range(20):
    e = exp.generateExpression()
    print("".join(e))
    oh = exp.to_one_hot(e)
    print(oh)
    print("".join(exp.to_character(oh)))