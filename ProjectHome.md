A collection of simple, minimal and self-contained code templates to be used as a base for creating new projects or prototypes.  Implementations are usually provided in OCaml, C and Python (in order of my preference).

All the examples are free of charge, distributed with the MIT license.  You can freely copy&paste this code into your own projects.  If you find any of these samples useful or you have code to contribute, please drop me an e-mail at `jjhellst@gmail.com`.

Examples of graphics templates:

  * How to open a window and render a pixel on it
  * How to render a triangle using OpenGL
  * How to render into a array of pixels and display the result on the screen
  * How to save the contents of framebuffer into a .png or .tga

| ![http://aihiot.googlecode.com/svn/trunk/gfx/draw_triangle/ocaml/screenshot_triangle.png](http://aihiot.googlecode.com/svn/trunk/gfx/draw_triangle/ocaml/screenshot_triangle.png) | ![http://aihiot.googlecode.com/svn/trunk/gfx/draw_bitmap/glut/draw_bitmap_screenshot.png](http://aihiot.googlecode.com/svn/trunk/gfx/draw_bitmap/glut/draw_bitmap_screenshot.png) |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

A few guiding principles for the creation of such examples:

  * None of the templates should depend on a common library.  I.e., each of them have to be self-contained even if it would lead to code duplication.  This is to make them easily copy&pasteable into your own projects.