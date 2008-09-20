
module I32 = Int32

let (>>) = I32.shift_right_logical
let (<<) = I32.shift_left
let (&) = I32.logand
let (^) = I32.logxor

let (|>) f g = g f

let crc_table =
  let tt = I32.of_string "0xedb88320" in
  let crc i =
    let t = ref (I32.of_int i) in
    for j = 0 to 7 do
      let t0 = 
        tt & (I32.add (I32.lognot (!t & I32.one)) I32.one) in
      t := (!t >> 1) ^ t0
    done;
    !t in
  let tbl = Array.init 256 crc in
  tbl
    
let acc_crc crc v =
  crc_table.(I32.to_int ((crc ^ (I32.of_int v)) & I32.of_int 0xff)) ^ (crc >> 8)

let acc_adler (a1,a2) v =
  let a1' = (a1 + v) mod 65521 in
  (a1', ((a2 + a1') mod 65521))

let write_dword chnl i =
  let output_byte_i32 i = 
    output_byte chnl (I32.to_int i) in
  output_byte_i32 (i >> 24);
  output_byte_i32 (i >> 16);
  output_byte_i32 (i >> 8);
  output_byte_i32 i
  
(** Save a .png file to `chnl'.  Note that chnl needs to be opened for
    binary output, otherwise output bytes will get messed up on
    Windows. *)
let write_png_chnl chnl pixels w h =
  let bpp = 4 in
  let idatlen = 6 + h * (6 + w * bpp) in
  Printf.fprintf chnl "\x89PNG\x0d\x0a\x1a\x0a\x00\x00\x00\x0dIHDR";
  let hdr = 
    [0; w lsr 16; w lsr 8; w;
     0; h lsr 16; h lsr 8; h;
     8; 6; 0; 0; 0] in

  let output_crc_byte crc v =
    let b = v land 255 in
    output_byte chnl b;
    acc_crc crc b in

  let output_adler_byte (crc,a1,a2) v =
    let b = v land 255 in
    output_byte chnl b;
    let (a1,a2) = acc_adler (a1,a2) b in
    (acc_crc crc v,a1,a2) in

  let crc = 
    List.fold_left
      (fun crc v -> output_crc_byte crc v)
      (I32.of_string "0x575e51f5") hdr in
  write_dword chnl (I32.lognot crc);
  write_dword chnl (I32.of_int idatlen);
  Printf.fprintf chnl "IDAT\x78\x01";

  let rec write_rows ((crc,adler1,adler2) as acc) y =
    if y < h then
      let blocklen = w * bpp + 1 in
      let bytes = [(if (y = h-1) then 1 else 0);
                   blocklen; blocklen lsr 8;
                   lnot blocklen; (lnot blocklen) lsr 8] in

      let rec output_row x acc =
        if x < w then
          let pix = pixels.(x+y*w) in
          let pix_bytes = [pix lsr 16; pix lsr 8; pix; 255] in
          output_row (x+1) 
            (List.fold_left (fun acc b -> output_adler_byte acc b) acc pix_bytes)
        else
          acc in

      let crc = 
        List.fold_left output_crc_byte crc bytes in
      let adler = 
        output_adler_byte (crc,adler1,adler2) 0 in
      write_rows (output_row 0 adler) (y+1)
    else 
      acc in
  
  let (crc,adler1,adler2) = 
    write_rows ((I32.of_string "0x13e5812d"),1,0) 0 in

  let bytes = [adler2 lsr 8; adler2; adler1 lsr 8; adler1] in
  let crc = List.fold_left output_crc_byte crc bytes in
  write_dword chnl (I32.lognot crc);
  write_dword chnl (I32.zero);
  Printf.fprintf chnl "IEND\xae\x42\x60\x82"

    
(** Save a Targa (.png) file to file `filename'. *)
let write_png filename pixels w h =
  let chnl = open_out_bin filename in
  try
    write_png_chnl chnl pixels w h;
    close_out chnl
  with
    _ -> 
      close_out chnl
