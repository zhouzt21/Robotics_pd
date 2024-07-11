import os, sys

import curses, atexit
#window: curses.I = None
window = None

special_keys = {
    259: 'up', 258: 'down',
    261: 'right', 260: 'left',
}


def init_window():
    global window
    if window is not None:
        return
    window = curses.initscr()
    window.keypad(True)
    curses.noecho()
    curses.cbreak()

    # Without this the program works fine, but the terminal remains borked
    # after exit. If this happens to you, run `reset` to restore (just type and
    # press enter, even if you don't see the text).
    atexit.register(curses.endwin)
    atexit.register(curses.nocbreak)
    atexit.register(curses.echo)

# Note: window.getch() returns int, not str.
def _get_key():
    init_window()
    keycode = window.getch()
    if keycode > 256:
        return special_keys[keycode]
    else:
        return chr(keycode)

def _display(text):

    init_window()
    window.addstr(0, 0, text)
    window.clrtobot()
    window.refresh()

current_text = ''


def display(text, mode=None):
    """
    Clears the screen and refills it with the given text.
    """
    while not isinstance(text, str):
        text = ''.join(text)

    global current_text
    current_text = text

    if mode == 'text':
        return text
    return _display(text)

def set_display(line, column, text, mode=None):
    """
    Changes only a portion of the display, keeping the rest constant.
    This function assumes the display has been set by the function `display`.
    Multi-character replacements are supported, but multi-line ones not.
    """
    lines = current_text.split('\n')
    x, y = column, line
    lines[y] = lines[y][:x] + text + lines[y][x + len(text):]
    return display('\n'.join(lines), mode)


"""
Map of keys and functions. Every time one of these keys is pressed, the
respective function will be called. The pressed key is not returned by
get_key() or similar.
"""
hotkeys = {}

def get_key():
    """
    Waits for user keyboard input and returns the character typed, or special
    key such as "up".
    """
    while True:
        key = _get_key()
        if key in hotkeys:
            hotkeys[key]()
            continue
        else:
            return key

def get_valid_key(expected_keys):
    """
    Waits until the user presses one of the keys in `expected_keys` and returns
    it.
    """
    while True:
        key = get_key()
        if key in expected_keys:
            return key

def get_option(option_by_key):
    """
    Given a map key -> option, waits until the user selects a valid key and
    then returns the associated option.
    """
    return option_by_key[get_valid_key(option_by_key)]

def process_input(function_by_key):
    """
    Given a map key -> function, loops receiving user input and invoking the
    respective functions. Unknown keys are ignored.
    To exit the loop, raise an exception from the invoked function.
    """
    while True:
        get_option(function_by_key)()

if __name__ == '__main__':
    while True:
        key = get_key()
        print(key)
        if key == 'q':
            exit()