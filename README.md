# CuttleShuttle-Analysis
Analysis code for the Cuttle Shuttle video dataset. 
Requires Visual Studio and Bonsai. 

## Video dataset 
[CuttleShuttle-VideoDataset-Raw](https://www.dropbox.com/sh/ep5j63nyx1by4tf/AAAJPIddR6b8YR787KuizMQya?dl=0)

## Video annotator
Open the Visual Studio solution "Bonsai.VideoAnnotations.sln" found in /Bonsai.VideoAnnotations/. Press F5 or hit the `Start` button. 
A new instance of Bonsai will open. From inside this instance, open the Bonsai workflow "CuttleShuttle-VideoAnnotator.bonsai" found in /Workflows/ShaderNavigator/. 

This workflow was used to hand-annotate the location of the center of the cuttlefish's mantle and the tip of its tail, then save a [snapshot](https://www.dropbox.com/sh/hlrvut5y4v0wqau/AABfFOVhv5kH2ZYa_FIozr-Pa?dl=0) of the annotated frame. 

When changing the video file you wish to annotate or view, make sure to check all nodes labeled "VideoFile", "VideoNavigator", "String", and "CsvReader" to make sure the file names are consistent. 

This workflow can also be used to visualize previous annotations and add more annotations. 

## Training dataset for machine learning
[Frame snapshots](https://www.dropbox.com/sh/hlrvut5y4v0wqau/AABfFOVhv5kH2ZYa_FIozr-Pa?dl=0) and [annotations](https://github.com/everymind/CuttleShuttle-Analysis/tree/master/Workflows/ShaderNavigator/annotations).
 
 

