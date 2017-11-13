# coding: utf-8

import random
import logging


def random_gen_func(allowed_var, allowed_mark, allowed_num, max_length):
    function = ''
    n = random.randint(1, max_length)
    state = random.randint(1, 2)

    for i in range(n):
        # print(state)
        symbol = ''
        if state == 1:
            state = 3
            symbol = random.choice(allowed_var)
        elif state == 2:
            state = random.choice([2, 3])
            symbol = random.choice(allowed_num)
        else:
            state = random.choice([1, 2])
            symbol = random.choice(allowed_mark)
        # print(symbol)
        function += symbol
    return function


def test_random(log_file_name, correct_expression, allowed_var, allowed_mark, allowed_num, max_length):
    n = 0
    logfile = open('log/' + log_file_name, 'w')
    while True:
        ans = random_gen_func(allowed_var, allowed_mark,
                              allowed_num, max_length)
        n += 1
        # logfile.write(ans + '\n')
        # print(ans)
        if correct_expression == ans:
            logfile.write(ans + '\n')
            break
        if (n % 100000 == 0):
            logfile.write(ans + '\n')
            print(n)

    logfile.write('result: ' + str(n))
    logfile.close()
    return n


ALLOWED_PARAMETERS = list('XY')
ALLOWED_SYMBOLS = ALLOWED_PARAMETERS + list('0123456789+-*/')

#

var = list('XY')
symbol = ['+', '-', '*', '/', '**']
num = list('0123456789')

# a = random_gen_func(var, symbol, num, 5)
n = 4
while True:
    test_random('random_rule' + str(n) + '.log',
                '3*X+2*Y', var, symbol, num, 7)
# print(a)
