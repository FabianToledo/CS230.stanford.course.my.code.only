
I did not uploaded the yolo .h5 model file because is ~200Mb long.
It can be re-generated cloning the yad2k project from: https://github.com/allanzelener/YAD2K
Here are the steps I did to re-generate it:

1 - Download the yolov2.weights from 
https://pjreddie.com/media/files/yolov2.weights

2 - Download the yolov2.cfg file from:
https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg 
 Note: Another way to get the cfg file is 
 cloning the repo from: https://github.com/pjreddie/darknet
 the yolov2.cfg is in the /cfg folder.

3 - Ejecute the python file yad2k.py with the following arguments
python yad2k.py yolo.cfg yolo.weights model_data/yolov2.h5

 Note: I had to update the function space_to_depth_x2 for the library to work because
       in the Tensorflow version I have (2.4.0) space_to_depth was moved to tensorflow.nn
 
 def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.nn.space_to_depth(x, block_size=2) # <-- old line: return tf.space_to_depth(x, block_size=2)
