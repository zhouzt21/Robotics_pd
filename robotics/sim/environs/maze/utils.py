import cv2
import numpy as np


def get_maze_env_obs(maze, dim=6):
    mmap = maze.maze
    height, width = mmap.height, mmap.width

    obs = np.zeros((dim, height, width))

    for i in range(width):
        for j in range(height):
            cell = mmap[i, j]
            if 'n' in cell: obs[0, j, i] = 1
            if 's' in cell: obs[1, j, i] = 1
            if 'w' in cell: obs[2, j, i] = 1
            if 'e' in cell: obs[3, j, i] = 1
    return obs

def render_background(maze, bs):
    mmap = maze.maze
    height, width = mmap.height, mmap.width
    background = np.zeros((bs * height, bs * width, 3), dtype=np.uint8) + np.array((128, 0, 128), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            cell = mmap[i, j]
            lines = []
            if 'n' in cell:
                lines.append([(i, j), (i + 1, j)])
            if 's' in cell:
                lines.append([(i, j + 1), (i + 1, j + 1)])
            if 'w' in cell:
                lines.append([(i, j), (i, j + 1)])
            if 'e' in cell:
                lines.append([(i + 1, j), (i + 1, j + 1)])
            for a, b in lines:
                a = (a[0] * bs, a[1] * bs)
                b = (b[0] * bs, b[1] * bs)
                cv2.line(background, a, b, (128, 255, 0), 2)
    return background


def draw_point(img, pos, rgb, bs, radius=3):
    if pos is None:
        return
    x, y = map(int, np.array(pos)*bs)
    cv2.circle(img, (x, y), radius, rgb, -1)

def set_block_color(img, pos, rgb, bs):
    if pos is None:
        return
    x, y = pos
    if x * bs < img.shape[1] and x>=0 and y * bs <img.shape[0] and y>=0:
        x, y = map(int, pos)
        xx = img[y * bs:(y + 1) * bs, x * bs:(x + 1) * bs] == np.array((128, 0, 128))
        xx = xx.all(axis=-1)
        img[y * bs:(y + 1) * bs, x * bs:(x + 1) * bs][xx] = rgb
    return img

def render_maze(maze, block_size, _background,
                loc, goal, subgoal, mode='human'):
    if mode == 'human':
        maze.render()
    elif mode == 'text':
        return maze.render(mode)
    elif mode == 'rgb_array' or mode == 'plt':
        img = _background.copy()
        draw_point(img, loc, (255, 255, 255), block_size)
        #draw_point(img, goal, (255, 255, 0), block_size)
        if len(subgoal) == 2:
            if tuple(goal) != tuple(subgoal):
                set_block_color(img, goal, (255, 255, 0), block_size)
                set_block_color(img, subgoal, (0, 0, 255), block_size)
            else:
                set_block_color(img, subgoal, (255, 255, 255), block_size)
        else:
            set_block_color(img, goal, (255, 255, 0), block_size)
            #print(subgoal)
            subgoal = subgoal[:2] * block_size
            cv2.circle(img, (int(subgoal[0]), int(subgoal[1])), 2, (0, 0, 255), -1)

        if mode == 'plt':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        return img
    else:
        raise NotImplementedError




def line_intersection(line1, line2):
    #calculate the intersection point
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0]
             [1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def check_cross(x0, y0, x1, y1):
    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    return np.cross(x1-x0, y0-x0), np.cross(y0-x0, y1-x0)


def check_itersection(x0, y0, x1, y1):
    EPS = 1e-10

    def sign(x):
        if x > EPS:
            return 1
        if x < -EPS:
            return -1
        return 0

    f1, f2 = check_cross(x0, y0, x1, y1)
    f3, f4 = check_cross(x1, y1, x0, y0)
    if sign(f1) == sign(f2) and sign(f3) == sign(f4) and sign(f1) != 0 and sign(f3) != 0:
        return True
    return False


def rect_lines(rect):
    (x0, y0), (x1, y1) = rect
    yield (x0, y0), (x1, y0)
    yield (x1, y0), (x1, y1)
    yield (x1, y1), (x0, y1)
    yield (x0, y1), (x0, y0)

def step(rects, x, y, dx, dy, boundary=0.00001):
    state = np.array([x, y])
    dest = np.array((x + dx, y + dy))

    min_dist = np.linalg.norm(state - dest)
    nearest = dest
    line = (state, dest)
    dir = (dest - state)/min_dist

    for i in rects:
        for l in rect_lines(i):
            if check_itersection(state, dest, l[0], l[1]):
                intersection = line_intersection(line, l)
                d = np.linalg.norm(state - intersection)
                if d < min_dist:
                    min_dist = d - boundary
                    nearest = state + min_dist * dir
    return nearest