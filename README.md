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
By simply running the train_lakeice.sh, the training will start.
For parameters: the specified values were used for all experiments.
1. Setup up the tensorflow records in LAKEICE_DATASET parameter.
2. --model_variant="xception_65" -> Change to "xception_65_skips" to use Deep-U-Lab  
   --skips=0                     -> Change to 1, if using "xception_65_skips"  
   --atrous_rates=6   
   --atrous_rates=12   
   --atrous_rates=18    
   --output_stride=16   
   --decoder_output_stride=4   
   --train_crop_size="321,321"   -> Used 512,512 for lake-detection and 321,321 for lake-ice segmentation  
   --dataset="lake"               
   --train_batch_size=8          -> Set according to GPU availability. This should be >=16 for tuning the batch norm layers  
   --training_number_of_steps="${NUM_ITERATIONS}"    
   --fine_tune_batch_norm=false  -> Set to "true" if train_batch_size>=16      
   --train_logdir="${TRAIN_LOGDIR}"    
   --base_learning_rate=0.0001    
   --learning_policy="poly"        
   --tf_initial_checkpoint="/home/pf/pfshare/data/MA_Rajanie/pretrained/deeplabv3_pascal_trainval/model.ckpt"       
   --dataset_dir="${LAKEICE_DATASET}"     
   

For evaluation and visualization, run the eval_lakeice.sh script.  
  
   --eval_split="val"             -> Split should be "val", instead of "train"     
   --model_variant="xception_65"  -> Same rules as train script
   --skips=0    
   --eval_crop_size="325,1210"    -> Full image eval_crop_size   
   --max_number_of_evaluations=1  -> If set to 1, evaluation script will run once and exit. If >1, it will keep checking the train logdir for new checkpoints. Useful, when running both train and eval scripts simultaneously (alloting part of GPU to both).    

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
