import tensorflow as tf

model = tf.keras.models.load_model("nn_model.h5")
model.save("nn_model.keras")

print("Model converted successfully!")
