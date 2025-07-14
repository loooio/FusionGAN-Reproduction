Tensorflow   2.18.0
h5py			   3.14.0
imageio			 2.36.0
matlablib   		1.2
numpy  	   		1.26.1
opencv-python		4.7.0.72
pillow			 23.2.1
scipy		     1.15.0

代码调整：
将所有 tf 调用替换为 tf.compat.v1 兼容版本
添加了 tf.compat.v1.disable_eager_execution() 以禁用 TensorFlow 2.x版本 的 eager execution
将 tf.variable_scope 替换为 tf.compat.v1.variable_scope
将 tf.get_variable 替换为 tf.compat.v1.get_variable
将 tf.truncated_normal_initializer 替换为 tf.compat.v1.truncated_normal_initializer
将 tf.random_uniform 替换为 tf.random.uniform
将 tf.contrib.layers.batch_norm 替换为 tf.compat.v1.layers.batch_normalization
将 tf.Session() 替换为 tf.compat.v1.Session(config=config)
将 tf.global_variables_initializer() 替换为 tf.compat.v1.global_variables_initializer()
将 tf.summary 替换为 tf.compat.v1.summary
将 xrange 替换为 range 
导入 imageio 替代 scipy.misc.imread 
使用 mode='F' 替代 flatten=True 用于灰度图像
使用 np.float32 替代 np.float
使用 tf.keras.layers.BatchNormalization 替代 tf.contrib.layers.batch_norm
