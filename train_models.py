import tensorflow as tf
from urllib.request import urlopen
import numpy as np
import os
import time

BATCH_SIZE = 64

def process_text():
	url = 'https://www.gutenberg.org/files/174/174.txt'
	text = urlopen(url).read().decode(encoding='utf-8')
	# length of text is the number of characters in it
	print ('Length of text: {} characters'.format(len(text)))

	# The unique characters in the file
	vocab = sorted(set(text))
	print ('{} unique characters'.format(len(vocab)))

	# Creating a mapping from unique characters to indices
	char2idx = {u:i for i, u in enumerate(vocab)}
	idx2char = np.array(vocab)

	text_as_int = np.array([char2idx[c] for c in text])
	return char2idx, idx2char



def generate_text(model, start_string):
  # Get Text vars:
  char2idx, idx2char = process_text()

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

def main():
	print('LOADING WEIGHTS....')
	model = tf.keras.models.load_model('./dorian_model')
	print('GENERATING TEXTs')
	print(generate_text(model, start_string=u"A million words are worth it. "))

if __name__ == '__main__':
	main()

