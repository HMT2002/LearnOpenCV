import numpy as np
import tensorflow as tf

# Create the interpreter and get the signature runner.
interpreter = tf.lite.Interpreter(
    model_path='movinet_a0_stream_k600_int8.tflite')

runner = interpreter.get_signature_runner()
input_details = runner.get_input_details()

def quantized_scale(name, state):
  """Scales the named state tensor input for the quantized model."""
  dtype = input_details[name]['dtype']
  scale, zero_point = input_details[name]['quantization']
  if 'frame_count' in name or dtype == np.float32 or scale == 0.0:
    return state
  return np.cast((state / scale + zero_point), dtype)

# Create the initial states, scale quantized.
init_states = {
    name: quantized_scale(name, np.zeros(x['shape'], dtype=x['dtype']))
    for name, x in input_details.items()
    if name != 'image'
}

# Insert your video clip or video frame here.
# Input to the model be of shape [1, 1, 172, 172, 3].
video = np.ones([1, 50, 172, 172, 3], dtype=np.float32)
frames = np.split(video, video.shape[1], axis=1)

# To run on a video, pass in one frame at a time.
states = init_states
for frame in frames:
  # Normally the input frame is normalized to [0, 1] with dtype float32, but
  # here we apply quantized scaling to fit values into the quantized dtype.
  frame = quantized_scale('image', frame)
  # Input shape: [1, 1, 172, 172, 3]
  outputs = runner(**states, image=frame)
  # `logits` will output predictions on each frame.
  logits = outputs.pop('logits')
  states = outputs