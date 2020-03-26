# DiceGame

#### The Game
Check if we have a good chance of winning in a game with two dices. 

The values of the dices are added. We start the game with 50c. The profit is computed with to following table:

|Sum|Payback|Profit
|--- |--- |---
|12 |4 x input |+1,50 EUR
|11 |3 x input |+1,00 EUR
|10 |2 x input |+0,50 EUR
|7, 8, 9 |input back |+0,00 EUR
|2, 3, 4, 5, 6 |none |-0,50 EUR

Is it good to take part in this game? Try in a loop with 1000 iterations, if we lose or win
in the long run.
