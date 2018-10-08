# Binary-image-classification-with-CNN-s

A convolutional neural network is trained on labeled images to recognize and predict whether a particular object is shown on an unlabeled image. Although the presented network can learn on any kind of image and number of image classes, I use it here to distinguish cats from dogs. The dataset consists of 25.000 images of dogs and cats for training and testing. The dataset was obtained from https://www.kaggle.com/c/dogs-vs-cats.

The model uses tensorflow Dataset API as input pipeline, with the raw images and labeles encoded in a binary format called TFRecords.
Due to bad hardware only a few filters with shallow kernel depths could be used during the convolution. Still only after one epoch an accuracy of 80 % can be achieved.

    epoch_nr: 0, iter_nr: 50/750, acc. : 0.52, train_loss: 22.807
    epoch_nr: 0, iter_nr: 100/750, acc. : 0.57, train_loss: 21.984
    epoch_nr: 0, iter_nr: 150/750, acc. : 0.66, train_loss: 21.425
    epoch_nr: 0, iter_nr: 200/750, acc. : 0.67, train_loss: 20.734
    epoch_nr: 0, iter_nr: 250/750, acc. : 0.68, train_loss: 20.435
    epoch_nr: 0, iter_nr: 300/750, acc. : 0.69, train_loss: 19.511
    epoch_nr: 0, iter_nr: 350/750, acc. : 0.69, train_loss: 19.563
    epoch_nr: 0, iter_nr: 400/750, acc. : 0.72, train_loss: 18.973
    epoch_nr: 0, iter_nr: 450/750, acc. : 0.72, train_loss: 18.387
    epoch_nr: 0, iter_nr: 500/750, acc. : 0.74, train_loss: 17.835
    epoch_nr: 0, iter_nr: 550/750, acc. : 0.72, train_loss: 18.457
    epoch_nr: 0, iter_nr: 600/750, acc. : 0.70, train_loss: 17.701
    epoch_nr: 0, iter_nr: 650/750, acc. : 0.74, train_loss: 17.824
    epoch_nr: 0, iter_nr: 700/750, acc. : 0.74, train_loss: 16.662
    epoch_nr: 1, iter_nr: 50/750, acc. : 0.78, train_loss: 17.079
    epoch_nr: 1, iter_nr: 100/750, acc. : 0.77, train_loss: 16.507
    epoch_nr: 1, iter_nr: 150/750, acc. : 0.78, train_loss: 16.892
    epoch_nr: 1, iter_nr: 200/750, acc. : 0.75, train_loss: 15.904
    epoch_nr: 1, iter_nr: 250/750, acc. : 0.76, train_loss: 15.442
    epoch_nr: 1, iter_nr: 300/750, acc. : 0.78, train_loss: 15.647
    epoch_nr: 1, iter_nr: 350/750, acc. : 0.77, train_loss: 15.200
    epoch_nr: 1, iter_nr: 400/750, acc. : 0.75, train_loss: 15.328
    epoch_nr: 1, iter_nr: 450/750, acc. : 0.79, train_loss: 15.536
    epoch_nr: 1, iter_nr: 500/750, acc. : 0.78, train_loss: 14.382
    epoch_nr: 1, iter_nr: 550/750, acc. : 0.75, train_loss: 15.656
    epoch_nr: 1, iter_nr: 600/750, acc. : 0.78, train_loss: 14.744
    epoch_nr: 1, iter_nr: 650/750, acc. : 0.78, train_loss: 13.968
    epoch_nr: 1, iter_nr: 700/750, acc. : 0.80, train_loss: 13.781
    epoch_nr: 2, iter_nr: 50/750, acc. : 0.80, train_loss: 14.569
    epoch_nr: 2, iter_nr: 100/750, acc. : 0.77, train_loss: 13.721
    epoch_nr: 2, iter_nr: 150/750, acc. : 0.79, train_loss: 14.982
    epoch_nr: 2, iter_nr: 200/750, acc. : 0.81, train_loss: 13.918
