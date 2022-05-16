import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

songs = mdl.lab1.load_training_data()

example_song = songs[0]

songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
print("There are ", len(vocab), " unique characters in the dataset")

# define a numerical representation
char2idx = {u : i for i, u in enumerate(vocab)}

# mapping from number back to character
idx2char = np.array(vocab)

# see the mapping
print('{')
for char,_ in zip(char2idx, range(30)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# function to vectorize songs
def vectorize_string(song):
    song_vec = np.array([char2idx[i] for i in song])
    return song_vec

# vectorize the combined songs
vectorized_songs = vectorize_string(songs_joined)

# check the mapping (repr makes it so that \n is printed instead of a newline character)
print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))

# want to train the computer so that it can correctly guess the next character
# the inputs will have a length of seq_length and the target will be the same length
# from the same text but shifted one character to the right.
# To do this, break the text into lengths seq_legnth + 1. For example,
# if seq_length is 4 and text is Hello, input will be Hell and target is ello
def get_batch(vectorized_songs, seq_length, batch_size):
    # n is the length of the songs
    n = vectorized_songs.shape[0] - 1

    # get random starting indice
    idx = np.random.choice(n - seq_length, batch_size)

    # list of input sequences
    input_batch = [vectorized_songs[idx[i] : idx[i] + seq_length] for i in range(batch_size)]

    # list of output sequences
    output_batch = [vectorized_songs[idx[i] + 1 : idx[i] + 1 + seq_length] for i in range(batch_size)]
    
    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

# RNN based off of LSTM architecture
# There is a state vector to maintain information about relationships between consecutive characters
# Output of LSTM is fed into a Dense softmax layer, which gives a distribution used to predict the next character
def LSTM(rnn_units) :
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True
    )

# The model will have 3 layers
# The first layer is embedding layer to transform indices to dense vectors
# The second layer is the LSTM
# The third layer is the Dense layer
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model

model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
model.summary()

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

# Checking the first example in the batch
sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices).numpy()
print(sampled_indices)
# Change indices to text
print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

# Training the model
# Using sparse_categorical_crossentropy loss (uses integer targets)
# Loss will be computed using true targets (labels) and predicted targets (logits)
def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)
    return loss
# Check loss of untrained model
example_batch_loss = compute_loss(y, pred)
print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())

# Parameters
num_training_iterations = 2000
batch_size = 32
seq_length = 250
learning_rate = 0.01

vocab_size = len(vocab)
embedding_dim = 512
rnn_units = 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

# # Using GradientTape to perform backpropagation
# model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)

# optimizer = tf.keras.optimizers.Adam(learning_rate)

# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         # Generate predictions using current model
#         y_hat = model(x)

#         # Compute loss
#         loss = compute_loss(y, y_hat)

#     # Compute gradients
#     grads = tape.gradient(loss, model.trainable_variables)

#     # Apply gradients to optimizer so it updates the model
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loss

# # Begin training
# history = []
# for x in range(num_training_iterations):
#     x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
#     loss = train_step(x_batch, y_batch)
#     history.append(loss.numpy().mean())

#     print(x)
#     # Save checkpoints
#     if x % 100 == 0:
#         model.save_weights(checkpoint_prefix)

# model.save_weights(checkpoint_prefix)

# # Plot loss
# plt.plot(history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.show()

# Use the model and restore weights to make music 
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()

# Generate music
def generate_text(model, start_string, generation_length = 1000):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for i in range(generation_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))

generated_text = generate_text(model, start_string="X", generation_length=2000)
print(generated_text)
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs): 
  # Synthesize the waveform from a song
  waveform = mdl.lab1.play_song(song)

  # If its a valid song (correct syntax), lets play it! 
  if waveform:
    print("Generated song", i)
    ipythondisplay.display(waveform)