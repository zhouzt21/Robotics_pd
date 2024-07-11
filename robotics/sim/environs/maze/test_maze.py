from rpg_envs.pacman.maze import MazeGame

#%%
a = MazeGame(7, 7)
a.reset()
print(a.player)

import sys
print(a.render(mode='text'))

a.step('down')
print(a.render(mode='text'))
