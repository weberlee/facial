import sys,tty,termios
import sys

class _Getch:
    """Gets a single character from standard input.
    Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

def get():
    inkey = _Getch()
    while(1):
        k=inkey()
        print("input::: ", k)
        if k != '':
            break
        if k == '\x1b[A':
            print("up")
        elif k == '\x1b[B':
            print("down")
        elif k == '\x1b[C':
            print("right")
        elif k == '\x1b[D':
            print("left")
        else:
            print("not an arrow key!")

def main():
    for i in range(0,20):
        get()

if __name__=='__main__':
    main()
