# coding: utf-8

import random
import logging


def random_gen_func(allowed_symbols, max_length):
    function = ''
    n = random.randint(1, max_length)
    for i in range(n):
        function += random.choice(allowed_symbols)
    return function


def test_random(log_file_name, correct_expression, allowed_symbols, max_length):
    n = 0
    logfile = open('log/' + log_file_name, 'w')
    while True:
        ans = random_gen_func(allowed_symbols, max_length)
        n += 1
        # logfile.write(ans + '\n')
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

test_random('random1.log', '3*X+2*Y', ALLOWED_SYMBOLS, 7)
