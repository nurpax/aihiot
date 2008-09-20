
(* Save the contents of a Graphics framebuffer to a Targa bitmap file. 

   Saving the .tga is intended to be free of external library
   dependencies to make it easy to grab into new projects.

   Code to actually save the Targa file is in Tga module.
*)

let save_screenshot fn =
  let w = Graphics.size_x () in
  let h = Graphics.size_y () in
  let img = Graphics.get_image 0 0 w h in
  let pixels = 
    Array.concat (List.rev (Array.to_list (Graphics.dump_image img))) in
  Tga.write_tga fn pixels w h

let _ = 
  (* First draw something on the screen.  Couple of triangles will
     do.  They should look like this /\/\ on the screen. *)
  Graphics.open_graph " 320x200";
  Graphics.set_color 0xff3322;
  Graphics.fill_poly [| (0,0); (80,200); (160,0) |];
  Graphics.set_color 0x3322ff;
  Graphics.fill_poly [| (160,0); (240,200); (320,0) |];
  save_screenshot "screenshot.tga";
  Printf.printf "Screenshot saved to file `screenshot.tga'.\n";
  flush_all ();
  while not (Graphics.button_down ()) do
    ()
  done
