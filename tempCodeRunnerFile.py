import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# List all GPUs detected
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detected!")
    for gpu in gpus:
        print("→", gpu)
else:
    print("❌ No GPU detected. Running on CPU.")

# Test if a simple operation runs on GPU
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])

