# Lake-Ice monitoring of Alpine lakes using webcam Images and crowdsourced data.

## What this repo contains?
1. Deeplab v3 plus tensorflow model adopted from official tensorflow repository with some changes.
  (a). Code for calculating Individual class IOU.
  (b). Code for checking confusion matrix on tensorboard.
  (c). Updated xception_65 model with extra skips from encoder to decoder.
2. Using labelme tool to create data annotations and code for converting json annotations to color-indexed masks.
3. Some data cleaning scripts (only valid for our lake-ice dataset).
4. Jupyter Notebook for visualizing data distribution for 5 classes : background, water, ice, snow, clutter.
5. Jupyter Notebook for inference using a saved tensorflow checkpoint.

## Steps to reproduce the experiment.

## Results and Visualisations.

## Beware of some common bugs.
1. for no modules called nets.
   Get the 'slim' directory from https://github.com/tensorflow/models/tree/master/research and from the research folder, run 
   ```python
   export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
   ```
2. Iterator end of line error.
   Look for empty lines in the dataset/"your dataset"/List/"train or val".txt files.
  
3. Dataset split  in train.py and eval.py, be careful to not use the default "trainval" split from original tensorflow deeplab    repository.

## References
1. Chen Liang-Chieh et. al 2018, Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, ECCV. https://github.com/tensorflow/models/tree/master/research/deeplab
    
2. Wad Ketaro 2016, labelme: Image Polygonal Annotation with Python. https://github.com/wkentaro/labelme
