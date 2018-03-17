import tensorflow as tf
import os


def _training_dataset(FLAGS):
    
    filenames=[FLAGS.root_dir_train+f for f in os.listdir(FLAGS.root_dir_train)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse,num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer)
    dataset = dataset.batch(FLAGS.batch_size_train)
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_train)
    
    return dataset
    
  
def _test_dataset(FLAGS):
    
    filenames=[FLAGS.root_dir_test+f for f in os.listdir(FLAGS.root_dir_test)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse,num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.batch(FLAGS.batch_size_val)
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_test)

    return dataset



def parse(serialized):
   
    features={'image/encoded': tf.FixedLenFeature([], tf.string),
              'image/label': tf.FixedLenFeature([], tf.int64),
              'image/height': tf.FixedLenFeature([], tf.int64),
              'image/width': tf.FixedLenFeature([], tf.int64),
              'image/num_channels': tf.FixedLenFeature([], tf.int64),
              }


    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    image_raw  = parsed_example['image/encoded']
    label = parsed_example['image/label']
    height = parsed_example['image/height']
    width = parsed_example['image/width']
    num_channels = parsed_example['image/num_channels']
    
    label = tf.cast(label, tf.int32)
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    num_channels = tf.cast(num_channels, tf.int32)
    
    image = tf.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)

    image=tf.reshape(image,(height,width,3))
    
    image=tf.image.resize_images(image,(64,64))
    
    
    

    return image, label