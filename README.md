The code related to the project is in the multi-snake directory. This code includes only the Variant 2 of the game, where the game continues even after one snake dies. The other variant can be easily obtained by modifying the terminating condition in multi-snake/snakeai/gameplay/environment.py.

To train the agents, execute:
python3 train.py --level <level-json-file.json> --num-episodes <num-episodes>

For e.g.
python3 train.py --level snakeai/levels/10x10-blank-2-snake.json --num-episodes 100000

To play the game, execute:
python3 play.py --interface <gui/cli> --level <level-json-file> --agent <agent-name> --num-episodes <num-episodes>

For e.g.
python3 play.py --interface gui --level snakeai/levels/10x10-blank-2-snake.json --agent minimaxdqn --num-episodes 100

Training the agents create some model files periodically which can be used for playing.
