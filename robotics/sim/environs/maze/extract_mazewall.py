from rpg_envs.pacman.maze.maze import MazeGame
import numpy as np

env = MazeGame(7, 7)

mmap = env.maze
height, width = mmap.height, mmap.width

obs = np.zeros((4, height, width))

scale = 1.

output = []
for i in range(width):
    for j in range(height):
        cell = mmap[i, j]
        x = i
        y = j
        if 'n' in cell and y > 0:
            output.append([[x, y], [x + 1, y]])
        if 'w' in cell and x > 0:
            output.append([[x, y], [x, y + 1]])

output += [
    [[0, 0], [0, height]],
    [[0, height], [width, height]],
    [[width, height], [width, 0]],
    [[width, 0], [0, 0]]
]

output = np.array(output) * 2 - 7
import json
#output.totxt
print(json.dumps(
    {'size': 7, 'walls': output.tolist()}
))

import torch
from rpg_envs.maze import LargeMaze
#print(output)

env = LargeMaze()
env.SIZE = 7
env.walls = torch.tensor(output)
print(env.walls.max(), env.walls.min())
img = env.render_wall()

import matplotlib.pyplot as plt
plt.imshow(img)
plt.savefig('test.png')