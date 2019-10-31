# Lake-Ice monitoring of Alpine lakes using webcam Images and crowdsourced data.

# What this repo contains?
1. Deeplab v3 plus tensorflow model adopted from official tensorflow repository with some changes.
  -Code for calculating Individual class IOU.
  -Code for checking confusion matrix on tensorboard.
  -Updated xception_65 model with extra skips from encoder to decoder.
2. Using labelme tool to create data annotations and code for converting json annotations to color-indexed masks.
3. Some data cleaning scripts (only valid for our lake-ice dataset).
4. Jupyter Notebook for visualizing data distribution for 5 classes : background, water, ice, snow, clutter.
5. Jupyter Notebook for inference using a saved tensorflow checkpoint.

# Beware of some common bugs.

# References
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}

@misc{labelme2016,
  author =       {Ketaro Wada},
  title =        {{labelme: Image Polygonal Annotation with Python}},
  howpublished = {\url{https://github.com/wkentaro/labelme}},
  year =         {2016}
}
