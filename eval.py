# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# """Evaluation script for the DeepLab model.

# See model.py for more details and usage.
# """



import os
import tensorflow as tf
import common
import model
from datasets import data_generator
import confusion_matrix
import my_metrics
from tensorboard import summary as summary_lib
import numpy as np
import keras.backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

#import wandb
#wandb.init(project="deeplab", sync_tensorboard=True)

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('eval_crop_size', '513,513',
                  'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 600 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.

flags.DEFINE_string('dataset', 'lake',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

flags.DEFINE_integer('skips', 0,
                     'Do you want extra skips layers from encoder to decoder')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.eval_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.eval_batch_size,
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      num_readers=2,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

  session_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)

  #session_config.gpu_options.allow_growth = True

   

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()
    #print(samples[common.IMAGE_NAME])

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.

    
    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])
    if tuple(FLAGS.eval_scales) == (1.0,):
  
      tf.logging.info('Performing single-scale test.')
      predictions, logits  = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid, skips=FLAGS.skips)
                                    
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')

      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          skips=FLAGS.skips,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))
    

    # weights = tf.to_float(tf.equal(labels, 0)) * 1 + \
    #           tf.to_float(tf.equal(labels, 1)) * 1 + \
    #           tf.to_float(tf.equal(labels, 2)) * 1 + \
    #           tf.to_float(tf.equal(labels, 3)) * 1 + \
    #           tf.to_float(tf.equal(labels, 4)) * 10 + \
    #           tf.to_float(tf.equal(labels, dataset.ignore_label)) * 0.0    

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    for eval_scale in FLAGS.eval_scales:
      predictions_tag += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:
      predictions_tag += '_flipped'

    # Define the evaluation metric.
  

    metric_map = {}

      # inserted to combat some error
    indices = tf.squeeze(tf.where(tf.less_equal(
        labels, dataset.num_of_classes - 1)), 1)
    labels_ind = tf.cast(tf.gather(labels, indices), tf.int32)
    predictions_ind = tf.gather(predictions, indices)
    # end of insert

    miou, update_miou = tf.metrics.mean_iou(
        labels_ind, predictions_ind,  dataset.num_of_classes, weights=weights, name="mean_iou")
    tf.summary.scalar(predictions_tag, miou)

    # Define the evaluation metric IOU for individual classes block starts
    iou_v, update_op = my_metrics.iou(
        labels_ind, predictions_ind, dataset.num_of_classes, weights=weights)
    for index in range(0, dataset.num_of_classes):
        metric_map['class_' + str(index) + '_iou'] = (iou_v[index], update_op[index])
        tf.summary.scalar('class_' + str(index) + '_iou', iou_v[index])

    ##auc curve
    
    # points, update_roc = tf.contrib.metrics.streaming_curve_points(labels=labels_ind, predictions=tf.reshape(logits, shape=[-1]),
    #                       weights=None, curve='ROC')
    # tf.summary.scalar("ROC_curve", points)


    ##

    #total_cm, update_cm =  _streaming_confusion_matrix(predictions_ind, labels_ind, num_classes=dataset.num_of_classes, weights=weights)
    #total_cm = tf.confusion_matrix(labels,predictions,num_classes=5,weights=None)
    #confusion = tf.Variable( tf.zeros([5,5], dtype=tf.int32 ), name='confusion' )
    confusionMatrixSaveHook = confusion_matrix.SaverHook(
        labels=['BG', 'water', 'ice', 'snow', 'clutter' ],
        confusion_matrix_tensor_name='mean_iou/total_confusion_matrix',
        summary_writer=tf.summary.FileWriterCache.get(str(FLAGS.eval_logdir))
    )

   
    summary_op = tf.summary.merge_all()

    summary_hook = tf.contrib.training.SummaryAtEndHook(
        log_dir=FLAGS.eval_logdir, summary_op=summary_op)
    hooks = [summary_hook, confusionMatrixSaveHook]

    # num_batches = int(
    #     math.ceil(len(samples) / float(FLAGS.eval_batch_size)))


    # tf.logging.info('Eval num images %d', len(samples))
    # tf.logging.info('Eval batch size %d and num batch %d',
    #                 FLAGS.eval_batch_size, num_batches)


    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations

    if FLAGS.quantize_delay_step >= 0:
      tf.contrib.quantize.create_eval_graph()

    tf.contrib.training.evaluate_repeatedly(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        eval_ops=[update_miou, update_op],
        max_number_of_evaluations=num_eval_iters,
        hooks=hooks,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  tf.app.run()
