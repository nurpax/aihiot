
(** Save a Targa (.tga) file to chnl. *)
let write_tga_chnl chnl pixels w h =
  let header = 
    [0; 0; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0;
     w land 255; w lsr 8;
     h land 255; h lsr 8; 32; 8] in
  assert (List.length header = 18);
  List.iter (fun e -> output_byte chnl e) header;
  for y = 0 to h-1 do
    for x = 0 to w-1 do
      let c = pixels.(x+(h-1-y)*w) in (* h-1-y = Flip image *)
      output_byte chnl (c land 255);
      output_byte chnl ((c lsr 8) land 255);
      output_byte chnl ((c lsr 16) land 255);
      output_byte chnl 255;
    done
  done
  
(** Save a Targa (.tga) file to file `filename'. *)
let write_tga filename pixels w h =
  let chnl = open_out_bin filename in
  try
    write_tga_chnl chnl pixels w h;
    close_out chnl
  with
    _ -> 
      close_out chnl
