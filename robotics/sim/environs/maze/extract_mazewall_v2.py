from rpg_envs.pacman.maze.maze import MazeGame
import numpy as np
from tools.utils import set_seed
set_seed(0)

env = MazeGame(5, 5)

mmap = env.maze
height, width = mmap.height, mmap.width

obs = np.zeros((4, height, width))

scale = 1.
gap = 0.2
havegap = True

output = []
for i in range(width):
    for j in range(height):
        cell = mmap[i, j]
        x = i
        y = j
        if y > 0:
            if 'n' in cell:
                output.append([[x, y], [x + 1, y]])
            elif havegap:
                output.append([[x, y], [x + 0.5-gap/2, y]])
                output.append([[x + 0.5+gap/2, y], [x+1, y]])
    
        if x > 0:
            if 'w' in cell:
                output.append([[x, y], [x, y + 1]])
            elif havegap:
                output.append([[x, y], [x, y + 0.5-gap/2]])
                output.append([[x, y+ 0.5+gap/2], [x, y+1]])
            

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