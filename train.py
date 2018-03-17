import tensorflow as tf
import time
from model import Model
import os
import numpy as np
from util import parse, _training_dataset, _test_dataset 


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir_train', 'C:/Users/Admin/Desktop/Deep Learning Local Datasets/Image Classification/Dogs vs Cats/tf_records_train/',
                           'Root directory of the training TFRecords'
                           )
tf.app.flags.DEFINE_string('root_dir_test', 'C:/Users/Admin/Desktop/Deep Learning Local Datasets/Image Classification/Dogs vs Cats/tf_records_test/',
                           'Root directory of the test TFRecords'
                           )
tf.app.flags.DEFINE_string('train_summary_path', 'C:/Users/Admin/Desktop/summary/train',
                           'Root directory of the test TFRecords'
                           )
tf.app.flags.DEFINE_string('test_summary_path', 'C:/Users/Admin/Desktop/summary/test',
                           'Root directory of the test TFRecords'
                           )
tf.app.flags.DEFINE_integer('num_train_tfrecords', 100,
                            'Number of the training TFRecrds files.'
                            )
tf.app.flags.DEFINE_integer('num_epoch', 100,
                            'Number of training epoch.'
                            )
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            'Learning rate.'
                          )
tf.app.flags.DEFINE_float('keep_prob', 0.9,
                          'Keeping probability for the neurons during drop out process.'
                          )
tf.app.flags.DEFINE_boolean('aplly_batch_norm', False,
                            'Whether to apply batch normalization of the conv layers.'
                            )
tf.app.flags.DEFINE_integer('batch_size_train', 32,
                            'Batch size of the training set.'
                            )
tf.app.flags.DEFINE_integer('batch_size_val', 500,
                            'Batch size of the validation set.'
                            )
tf.app.flags.DEFINE_integer('shuffle_buffer', 100,
                            'Buffer for the shuffeling of the data.'
                            )
tf.app.flags.DEFINE_integer('num_parallel_calls', 4,
                            'Number of parallel threads for the mapping function.'
                            )
tf.app.flags.DEFINE_integer('prefetch_buffer_train', 16,
                            'Buffer size of the prefetch for the training.'
                            )
tf.app.flags.DEFINE_integer('prefetch_buffer_test', 1,
                            'Buffer size of the prefetch for the testing.'
                            )
tf.app.flags.DEFINE_string('optimizer_type', 'adam',
                           'Kind of the optimizer for the training.'
                           )
tf.app.flags.DEFINE_integer('val_after', 50,
                            'Validate after number of iterations.'
                            )


  
def main(_):
    
    filenames=[FLAGS.root_dir_train+f for f in os.listdir(FLAGS.root_dir_train)] 
    num_batches=int(len(filenames*FLAGS.num_train_tfrecords)/FLAGS.batch_size_train)
     
    train_writer=tf.summary.FileWriter(FLAGS.train_summary_path)
    test_writer=tf.summary.FileWriter(FLAGS.test_summary_path)
    
    with tf.Graph().as_default():
        
        training_dataset=_training_dataset(FLAGS)
        validation_dataset=_test_dataset(FLAGS)
        
        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()
      
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, validation_dataset.output_types, validation_dataset.output_shapes)
        image, label = iterator.get_next()
        
        model=Model(FLAGS)
        
        training_op, loss_op, loss_summary = model.optimize(image,label)
        prediction_result, acc_summary=model._prediction_result(image, label)
  
        train_merged = tf.summary.merge([loss_summary])
        test_merged = tf.summary.merge([acc_summary])
              
        with tf.Session() as sess:
                
            sess.run(tf.global_variables_initializer())
            
            train_writer.add_graph(sess.graph)
            test_writer.add_graph(sess.graph)
                
            training_handle = sess.run(training_iterator.string_handle())
            validation_handle = sess.run(validation_iterator.string_handle())
          
            for epoch in range(0,FLAGS.num_epoch):
                    
                sess.run(training_iterator.initializer)
                temp_loss=0
                    
                for iter_nr in range(num_batches):
                        
                    _, l, summary=sess.run((training_op,loss_op,train_merged), feed_dict={handle: training_handle})
                    temp_loss+=l
                    train_writer.add_summary(summary, iter_nr)
                        
                    if iter_nr!=0 and iter_nr%FLAGS.val_after==0:
                            
                        sess.run(validation_iterator.initializer)
                        
                        acc, summary=sess.run((prediction_result,test_merged), feed_dict={handle: validation_handle})
                        test_writer.add_summary(summary, iter_nr)
                        
                        print('epoch_nr: %i, iter_nr: %i/%i, acc. : %.2f, train_loss: %.3f' %(epoch, iter_nr, num_batches,acc, (temp_loss/FLAGS.val_after)))
                        temp_loss=0
                        
                        
               
if __name__ == "__main__":
    
    tf.app.run()





        