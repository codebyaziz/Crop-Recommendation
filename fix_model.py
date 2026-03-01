import keras

print("Keras version:", keras.__version__)

model = keras.models.load_model("nn_model.keras")
model.save("nn_model.h5", include_optimizer=False)

print("Model converted successfully!")
