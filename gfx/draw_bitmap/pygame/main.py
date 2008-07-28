#!/usr/bin/env python

# On Debian/Ubuntu you may need to install pygame.  Fortunately it's as easy as:
#
#   apt-get install python-pygame
#
# Thanks to Simo Melenius for contributing this example.

import pygame, math
from pygame.locals import *

def setscreen ((width, height) = (256, 256)):
    """Update the window surface with WIDTH and HEIGHT."""
    global visual
    visual = pygame.display.set_mode((width, height), pygame.RESIZABLE)

def create_texture ():
    # Create a 256x256 texture
    texture = pygame.Surface((256, 256), depth=24)
	# Get a Surfarray out of it
    texture_array = pygame.surfarray.pixels3d(texture)
	# Manipulate the surfarray to render something on it
    for x in xrange(256):
        a = x * 0.1
        xcol = (math.sin(a)*0.5+0.5)*255.0
        for y in xrange(256):
            c = (xcol, y, 0)
            texture_array[x][y] = c
	# Garbage-collecting the surfarray will unlock the surface.
    return texture

#
# Main loop
#
pygame.init ()
setscreen()
texture = create_texture()
scaled_texture = None

try:
    while True:
        # Scale surface if size has changed
        if not scaled_texture or scaled_texture.get_size() != visual.get_size():
            scaled_texture = pygame.transform.scale(texture, visual.get_size())
        # Draw surface
        visual.blit(scaled_texture, (0, 0))

        # Handle UI events
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
