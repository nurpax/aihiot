
(* Draw a red triangle over the 400x400 screen using OCaml's Graphics
   module.  Wait until the user clicks on the window and exit.
   
   This is by far the simplest method of doing it and doesn't require
   installing any extra libraries.  Should work on both Win32 and
   Linux (X11). 
*)

let _ = 
  Graphics.open_graph " 200x200";
  Graphics.set_color 0xff3322;
  Graphics.fill_poly [| (0,0); (200,100); (0,200) |];
  while not (Graphics.button_down ()) do
    ()
  done
