# https://neurowhai.tistory.com/161
# 위 URL 참고 및 커스터마이징
from keras import layers, models
from keras import datasets
from keras import backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


# 학습 정보
batch_size = 64
epochs = 64
latent_dim = 256
num_samples = 1024 # 학습 데이터 개수
 
# 문장 벡터화
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
 
# 데이터 입력
lines = []
f = open('train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    print(line)
    lines.append(line)
f.close()





for line in lines[: min(num_samples, len(lines) - 1)]:
  input_text, target_text = line
  input_texts.append(input_text)
  target_texts.append(target_text)
 
  # 문자 집합 생성
  for char in input_text:
    if char not in input_characters:
      input_characters.add(char)
  for char in target_text:
    if char not in target_characters:
      target_characters.add(char)
 
num_samples = len(input_texts)
            
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
 
print('Number of samples:', num_samples)
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


print(input_characters)
print(target_characters)


# 문자 -> 숫자 변환용 사전
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
 
# 학습에 사용할 데이터를 담을 3차원 배열
encoder_input_data = np.zeros(
    (num_samples, max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
 
# 문장을 문자 단위로 원 핫 인코딩하면서 학습용 데이터를 만듬
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
  for t, char in enumerate(input_text):
    encoder_input_data[i, t, input_token_index[char]] = 1.
  for t, char in enumerate(target_text):
    decoder_input_data[i, t, target_token_index[char]] = 1.
    if t > 0:
      decoder_target_data[i, t - 1, target_token_index[char]] = 1.
      
# 인코더 생성
encoder_inputs = layers.Input(shape=(None, num_encoder_tokens))
encoder = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 인코더의 출력은 필요없고 상태만 중요
encoder_states = [state_h, state_c]
 
# 디코더 생성.
decoder_inputs = layers.Input(shape=(None, num_decoder_tokens))
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
# 디코더의 내부 상태는 학습에 사용하지 않지만 추론(테스트) 단계에서는 필요함.
# 디코더의 초기 상태를 인코더의 최종 상태로 설정.
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
 
# 학습
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          verbose=2)
