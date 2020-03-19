import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, BatchNormalization
#from embedding import one_hot_encoding

#[OMP: Error #15: Initializing libiomp5.dylib]에러 해결용
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class Char_RNN(object):
    def __init__(self, input_size, hidden_size, num_decoder, iter=100):
        self.hidden_size = hidden_size
        self.batch_size = input_size[0]
        self.time_step = input_size[1]
        self.n_class = input_size[2]
        self.num_decoder = num_decoder
        self.iter = iter
        self.build()

    def build(self):
        self.x_data = tf.placeholder(shape=[None, self.time_step, self.n_class], dtype=tf.float32)
        self.y_data = tf.placeholder(dtype=tf.int32, shape=[None, ])

        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.5)

        # cell, inputs, dtype
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.x_data, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        self.outputs = outputs[-1]
        print(self.outputs)
        

    def fit(self, train_data, target_data):
        weights = tf.Variable(tf.random_normal(shape=[self.hidden_size, self.n_class]))
        bias = tf.Variable(tf.random_normal(shape=[self.n_class]))
        logits = tf.matmul(self.outputs, weights) + bias

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_data))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        
        # Launch the graph in a session
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.iter):
                _, _loss, _logits = sess.run((optimizer, loss, logits), feed_dict={self.x_data: train_data, self.y_data: target_data})
                self.result = np.argmax(_logits, -1)
                accuracy = np.mean(self.result==target_data)
                if i % 50 == 0:
                    print("step: {}, loss: {:.3f}, acc: {:.2f}".format(i, _loss, accuracy))
    
    def decode(self, vocab_dict):
        decoded = []
        reverse_vocab = {id : w for w, id in vocab_dict.items()}
        for batch in self.result:
            decoded.append(reverse_vocab[batch])
        return decoded


class Seq2Seq_for_translation(object):
    def __init__(self, input_texts, target_texts, latent_dim=256):
        self.latent_dim = latent_dim

        input_characters = set()
        target_characters = set()
        for input_text, target_text in zip(input_texts, target_texts):
            for ch in input_text:
                if ch not in input_characters:
                    input_characters.add(ch)
            for ch in target_text:
                if ch not in target_characters:
                    target_characters.add(ch)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])

        self.input_token_index = { char: id for id, char in enumerate(input_characters)}
        self.target_token_index = { char: id for id, char in enumerate(target_characters)}

        self.encoder_input_data = np.zeros(shape=(len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
        self.decoder_input_data = np.zeros(shape=(len(target_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros(shape=(len(target_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for j, ch in enumerate(input_text):
                self.encoder_input_data[i, j, self.input_token_index[ch]] = 1.
            for j, ch in enumerate(target_text):
                self.decoder_input_data[i, j, self.target_token_index[ch]] = 1.
                if j > 0:
                    self.decoder_target_data[i, j-1, self.target_token_index[ch]] = 1.

        self.build()

    def build(self):
        # a part of encoder
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_input')
        encoder = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='encoder')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        # a part of decoder
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_input')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        batchNorm = BatchNormalization()
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(batchNorm(decoder_outputs))
               
        # a model to train
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # encoder model to decode
        self.encoder_model = Model(encoder_inputs, encoder_states)

        # decoder model to decode
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def fit(self, batch_size=64, epochs=20, optimizer='rmsprop', loss='categorical_crossentropy', load_model_path=None, save_model_path='s2s.h5'):
        if not load_model_path == None:
            load_model(load_model_path)
        self.model.compile(optimizer, loss)
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
            batch_size = batch_size,
            epochs=epochs,
            validation_split=0.2)
        self.model.save(save_model_path)

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors. [state_h, state_c]
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        # 점점 디코드된 문자열을 추가해나감.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
    
    def translate(self, input_seq):
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())
            
        input_vec = np.zeros(shape=(len(input_seq), self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
        for i, txt in enumerate(input_seq):
            for j, ch in enumerate(txt):
                input_vec[i, j, self.input_token_index[ch]] = 1.
        
        decoded_sequences = []
        for seq_idx in range(input_vec.shape[0]):
            decoded_sequences.append(self.decode_sequence(input_vec[seq_idx: seq_idx+1]))

        return decoded_sequences
    
    def draw(self):
        print('entire model:')
        print(self.model.summary())
        print('\nencoder model:')
        print(self.encoder_model.summary())
        print('\ndecoder model:')
        print(self.decoder_model.summary())


if __name__=="__main__":
    """
        # Data download
        English to French sentence pairs.
        http://www.manythings.org/anki/fra-eng.zip
        Lots of neat sentence pairs datasets can be found at:
        http://www.manythings.org/anki/
    """

    with open('kor-eng/kor.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    input_texts = []
    target_texts = []
    target_text = ""
    for line in lines[:3000]:
        input_text, target_text = line.split('\t')[:2]
        input_texts.append(input_text)
        # Each '\t' and '\n' is used as 'start' and 'end' sequence character
        target_text = '\t' + target_text + '\n'
        target_texts.append(target_text)

    seq2seq = Seq2Seq_for_translation(input_texts, target_texts)
    seq2seq.draw()
    seq2seq.fit(load_model_path='s2s.h5', epochs=10)
    print(input_texts[:100])
    print(seq2seq.translate(input_texts[:100]))
    while 1:
        sen = input("input : ")
        print("result : ",seq2seq.translate(sen))