"""
    The Dice Game module
"""

import random


def roll_dices_for_sum():
    """
    roll 2 dices and sum the numbers up, where each dice should yieds from 1 to 6

    Returns:
        result(int): the sum of all numbers
    """

    d1 = random.randint(1, 6)
    d2 = random.randint(1, 6)

    result = d1 + d2

    return result


def get_profit(the_sum: int):
    """
    get the profit according to the sum of all dices.

    - if 2 <= the_sum <= 6, profit is -0.5
    - if 7 <= the_sum <= 9, profit is 0
    - if the_sum == 10, profit is 0.5
    - if the_sum == 11, profit is 1.0
    - if the_sum == 12, profit is 1.5

    Args:
        the_sum(int): the sum of all dices, should between 2 to 12

    Returns:
        profit(float): the profit of the current round
    """

    if 2 <= the_sum <=6:
        profit = -0.5
    elif 7 <= the_sum <= 9:
        profit = 0
    elif the_sum == 10:
        profit = 0.5
    elif the_sum == 11:
        profit = 1.0
    elif the_sum == 12:
        profit = 1.5

    return profit


if __name__ == "__main__":

    balance = 0.5
    for i in range(5000):
        the_sum = roll_dices_for_sum()
        profit = get_profit(the_sum)
        balance = balance + profit

    print("After playing the game for 5000 rounds")

    _worthy = "worthy" if balance > 0.5 else "not worthy"
    print(f"The balance is {balance}, thus the game is {_worthy} to play.")
