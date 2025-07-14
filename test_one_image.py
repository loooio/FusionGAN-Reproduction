import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import glob
from PIL import Image
import time

# 配置TensorFlow（使用CPU多线程）
config = tf.compat.v1.ConfigProto(
    device_count={"CPU": 4},
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4,
)


def load_image(path, is_grayscale=True):
    img = Image.open(path)
    if is_grayscale:
        img = img.convert('L')
    return np.array(img).astype(np.float32)


def save_image(image, path):
    """保存图像并确保值范围正确"""
    # 先归一化到0-255范围
    if image.min() < 0 or image.max() > 255:
        image = (image - image.min()) * (255.0 / (image.max() - image.min()))
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def prepare_data(dataset):
    """准备数据，保持原始顺序"""
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp")) + glob.glob(os.path.join(data_dir, "*.jpg"))
    data.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return data

def lrelu(x,leak=0.2):
    return tf.maximum(x, leak*x)


def fusion_model(img):
    """图像融合模型"""
    with tf.compat.v1.variable_scope('fusion_model'):
        # Layer 1
        with tf.compat.v1.variable_scope('layer1'):
            weights = tf.constant(reader.get_tensor('fusion_model/layer1/w1'))
            bias = tf.constant(reader.get_tensor('fusion_model/layer1/b1'))
            conv1 = tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(conv1)
            conv1 = lrelu(conv1) # LeakyReLU

        # Layer 2
        with tf.compat.v1.variable_scope('layer2'):
            weights = tf.constant(reader.get_tensor('fusion_model/layer2/w2'))
            bias = tf.constant(reader.get_tensor('fusion_model/layer2/b2'))
            conv2 = tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(conv2)
            conv2 = lrelu(conv2)

        # Layer 3
        with tf.compat.v1.variable_scope('layer3'):
            weights = tf.constant(reader.get_tensor('fusion_model/layer3/w3'))
            bias = tf.constant(reader.get_tensor('fusion_model/layer3/b3'))
            conv3 = tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(conv3)
            conv3 = lrelu(conv3)

        # Layer 4
        with tf.compat.v1.variable_scope('layer4'):
            weights = tf.constant(reader.get_tensor('fusion_model/layer4/w4'))
            bias = tf.constant(reader.get_tensor('fusion_model/layer4/b4'))
            conv4 = tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv4 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(conv4)
            conv4 = lrelu(conv4)

        # Layer 5
        with tf.compat.v1.variable_scope('layer5'):
            weights = tf.constant(reader.get_tensor('fusion_model/layer5/w5'))
            bias = tf.constant(reader.get_tensor('fusion_model/layer5/b5'))
            conv5 = tf.nn.conv2d(conv4, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5 = tf.nn.tanh(conv5)

    return conv5


def process_image(ir_path, vi_path, sess, images_ir, images_vi, fusion_output, output_dir, index):
    """处理大尺寸图像对"""
    print(f"\nProcessing pair {index + 1}: {os.path.basename(ir_path)} and {os.path.basename(vi_path)}")
    start_time = time.time()

    # 加载原始尺寸图像
    ir_img = (load_image(ir_path) - 127.5) / 127.5  # 归一化到[-1,1]
    vi_img = (load_image(vi_path) - 127.5) / 127.5

    # 添加padding（根据您的模型需要6像素填充）
    padding = 6
    ir_img = np.pad(ir_img, ((padding, padding), (padding, padding)), 'edge')
    vi_img = np.pad(vi_img, ((padding, padding), (padding, padding)), 'edge')

    # 添加batch和channel维度
    ir_input = np.expand_dims(np.expand_dims(ir_img, 0), -1)
    vi_input = np.expand_dims(np.expand_dims(vi_img, 0), -1)

    print(f"Input shapes - IR: {ir_input.shape}, VI: {vi_input.shape}")

    fused = sess.run(fusion_output, feed_dict={
        images_ir: ir_input,
        images_vi: vi_input
    }).squeeze()

    # 值范围诊断
    print(f"Fused value range - min: {fused.min():.3f}, max: {fused.max():.3f}, mean: {fused.mean():.3f}")

    # 特殊处理灰色输出问题
    if np.allclose(fused.min(), fused.max(), atol=0.1):  # 如果动态范围太小
        print("Warning: Low dynamic range detected, applying contrast stretch")
        p2, p98 = np.percentile(fused, (2, 98))
        fused = np.clip((fused - p2) / (p98 - p2) * 255, 0, 255)
    else:
        fused = (fused * 127.5 + 127.5).clip(0, 255)  # 标准tanh反归一化

    # 保存结果
    output_path = os.path.join(output_dir, f'fused_{index}.bmp')
    save_image(fused, output_path)

    print(f"Saved to {output_path} | Size: {fused.shape} | Time: {time.time() - start_time:.2f}s")
    return output_path


def main():
    # 初始化
    global reader
    reader = tf.compat.v1.train.NewCheckpointReader('./checkpoint/CGAN_120/CGAN.model-3')

    # 准备数据
    data_ir = prepare_data('test_ir')
    data_vi = prepare_data('test_vi')

    # 检查两个文件夹图片数量是否相同
    assert len(data_ir) == len(data_vi), "Number of IR and VI images must match"
    num_images = len(data_ir)
    print(f"Found {num_images} image pairs to process")

    # 创建计算图
    with tf.compat.v1.name_scope('IR_input'):
        images_ir = tf.compat.v1.placeholder(tf.float32, [1, None, None, 1])
    with tf.compat.v1.name_scope('VI_input'):
        images_vi = tf.compat.v1.placeholder(tf.float32, [1, None, None, 1])
    input_image = tf.concat([images_ir, images_vi], axis=-1)
    fusion_output = fusion_model(input_image)

    # 输出目录
    output_dir = os.path.join(os.getcwd(), 'fusion_results_avesize')
    os.makedirs(output_dir, exist_ok=True)

    # 处理图像
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        results = []
        for i in range(num_images):
            try:
                result_path = process_image(
                    data_ir[i], data_vi[i], sess,
                    images_ir, images_vi, fusion_output,
                    output_dir, i
                )
                results.append(result_path)
            except Exception as e:
                print(f"Error processing pair {i}: {str(e)}")

    print("\nProcessing complete. Results:")
    for path in results:
        print(f"- {path}")

if __name__ == "__main__":
    main()