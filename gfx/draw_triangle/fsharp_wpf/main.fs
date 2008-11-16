#light

open System
open System.Collections.Generic 
open System.IO
open System.Windows
open System.Windows.Controls
open System.Windows.Markup
open System.Windows.Media
open System.Windows.Media.Media3D
open System.Xml

(* creates the window and loads given the Xaml file into it *)
let create_window (file : string) =
  using (XmlReader.Create(file))
    (fun (stream : XmlReader) -> 
       let temp = XamlReader.Load(stream) :?> Window in
       temp)

[<STAThread>]
do
  let window = create_window "MainWindow1.xaml" in
  let app = new Application() in 
  app.Run(window) |> ignore
