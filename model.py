import tensorflow as tf
import numpy as np



class Model:
    
    def __init__(self,FLAGS):
        
        self.FLAGS=FLAGS
        
        self.initializer=tf.random_normal_initializer(mean=0,stddev=0.02)
        self.bias_initializer=tf.zeros_initializer()
        
    
    def _inference(self, batch_x, isTraining):
        
        with tf.name_scope('conv_layers'):
        
            with tf.name_scope('conv_1_layer'):
                conv1=self._conv_layer(batch_x,filters=16, name='conv_1',is_training=isTraining)
                conv1=tf.nn.relu(conv1)
                conv1_dropout=tf.nn.dropout(conv1,self.FLAGS.keep_prob, name='conv_1_dropout')
            
            with tf.name_scope('conv_2_layer'):
                conv2=self._conv_layer(conv1_dropout,filters=16, name='conv2',is_training=isTraining)
                conv2=tf.nn.relu(conv2)
                conv2_dropout=tf.nn.dropout(conv2,self.FLAGS.keep_prob, name='conv_2_dropout')
            
            pool1=self._max_pool(conv2_dropout, name='pool_1')
        
            with tf.name_scope('conv_3_layer'):
                conv3=self._conv_layer(pool1,filters=32, name='conv_3',is_training=isTraining)
                conv3=tf.nn.relu(conv3)
                conv3_dropout=tf.nn.dropout(conv3,self.FLAGS.keep_prob, name='conv_3_dropout')
                
            with tf.name_scope('conv_4_layer'):     
                conv4=self._conv_layer(conv3_dropout,filters=32, name='conv4',is_training=isTraining)
                conv4=tf.nn.relu(conv4)
                conv4_dropout=tf.nn.dropout(conv4,self.FLAGS.keep_prob, name='conv_4_dropout')
                
            pool2=self._max_pool(conv4_dropout, name='pool_2')
            
            with tf.name_scope('conv_5_layer'): 
                conv5=self._conv_layer(pool1,filters=64, name='conv_5',is_training=isTraining)
                conv5=tf.nn.relu(conv5)
                conv5_dropout=tf.nn.dropout(conv5,self.FLAGS.keep_prob, name='conv_5_dropout')
                
            with tf.name_scope('conv_6_layer'):     
                conv6=self._conv_layer(conv5_dropout,filters=64, name='conv6',is_training=isTraining)
                conv6=tf.nn.relu(conv6)
                conv6_dropout=tf.nn.dropout(conv6,self.FLAGS.keep_prob, name='conv_6_dropout')
                
            pool2=self._max_pool(conv6_dropout, name='pool_2')
            
        with tf.name_scope('dense_layers'):    
        
            flat_conv_img=tf.contrib.layers.flatten(pool2)
            
            with tf.name_scope('dense_1_layer'):
                
                W1=tf.get_variable('W_1',shape=(flat_conv_img.shape[1],1024),initializer=self.initializer)
                b1=tf.get_variable('bias_1',shape=(1024),initializer=self.bias_initializer)
                a1=tf.nn.relu(tf.nn.bias_add(tf.matmul(flat_conv_img, W1), b1))
            
            with tf.name_scope('dense_2_layer'):
                W2=tf.get_variable('W_2',shape=(1024,2),initializer=self.initializer)
                b2=tf.get_variable('bias_2',shape=(2),initializer=self.bias_initializer)
                logits=tf.nn.bias_add(tf.matmul(a1, W2), b2)
            
        return logits
        

    def _compute_loss(self, logits, labels):
        
        with tf.name_scope('loss'):
            loss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                              logits=logits,
                                                                              name='cross_entropy'))
        loss_summary=tf.summary.scalar('cross_entropy', loss)  
        return loss, loss_summary
    
    
    def optimize(self, images,labels):
        
        with tf.variable_scope('inference'):
            logits = self._inference(images, isTraining=True)
            
        loss, loss_summary=self._compute_loss(logits, labels)
    
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            if self.FLAGS.optimizer_type=='adam':
                optimizer_op= tf.train.AdamOptimizer(self.FLAGS.learning_rate, name='adam_optimizer').minimize(loss)
            elif self.FLAGS.optimizer_type=='sgd':
                optimizer_op= tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate, name='sgd_optimizer').minimize(loss)
            else:
                raise ValueError('Not an avaiable optimizer!')
                
            return optimizer_op, loss,loss_summary
        
    def _prediction_result(self, images,labels):
        
        with tf.variable_scope('inference', reuse=True):
            
            logits=self._inference(images, isTraining=False)
        
        with tf.name_scope('accuracy'):
            
            prediction=tf.nn.softmax(logits)
            labels=tf.cast(labels, tf.int64)
            accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), labels), tf.float32))
            acc_summary=tf.summary.scalar('accuracy', accuracy)
            
        return accuracy, acc_summary
        

    def _conv_layer(self, x, filters, name, kernel_size=[3, 3],strides=(1, 1), batch_norm=True,is_training=True):
        
        conv=tf.layers.conv2d(x, filters=filters, 
                              kernel_size=kernel_size, 
                              strides=strides,
                              kernel_initializer=self.initializer,
                              padding='same',
                              activation=None,
                              name=name)
        
        if batch_norm==self.FLAGS.aplly_batch_norm:
            
            conv=tf.contrib.layers.batch_norm(conv, center=True, scale=True, 
                                              is_training=is_training,
                                              )
        return conv
    
    
    def _max_pool(self, x, name, pool_size=[2,2], strides=[2,2], padding='same'):
        
        return tf.layers.max_pooling2d(x, pool_size=pool_size,
                                       strides=strides, 
                                       padding=padding,name=name)
        
    
