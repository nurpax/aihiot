#!/usr/bin/env python
import pygame, math
from pygame.locals import *

# On Debian/Ubuntu you may need to install pygame.  Fortunately it's as easy as:
#
#   apt-get install python-pygame
#
# Thanks to Simo Melenius for contributing this example.

def setscreen ((width, height) = (200, 200)):
    """Update the window surface with WIDTH and HEIGHT."""
    global visual
    visual = pygame.display.set_mode((width, height), pygame.RESIZABLE)

#
# Main loop
#
pygame.init ()
setscreen()

try:
    while True:
        # Fill white and draw a red triangle.
        visual.fill((255, 255, 255))
        pygame.draw.polygon(visual, (255, 0, 0), [(0, 0),
                                                  (visual.get_width(), visual.get_height() / 2),
                                                  (0, visual.get_height())])

        # Handle UI events.
        for e in pygame.event.get ():
            if e.type == QUIT:
                raise KeyboardInterrupt
            elif e.type == VIDEORESIZE:
                setscreen(e.size)
            elif e.type == KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

        pygame.display.update ()
except KeyboardInterrupt:
    pass
