# -*- coding: utf-8 -*-
import numpy as np
import pickle
import urllib.request
from eunjeon import Mecab
from tensorflow import keras
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import ssl
context = ssl._create_unverified_context()

urllib.request.urlretrieve("https://raw.githubusercontent.com/thushv89/attention_keras/master/src/layers/attention.py", filename="attention.py")
from attention import AttentionLayer


config = tf.ConfigProto(
        device_count={'GPU' : 1},
        allow_soft_placement=True
)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6


graph = tf.get_default_graph()
session = tf.Session(config=config)

mecab = Mecab(dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic')
np.random.seed(seed=0)

# 전처리 함수
def preprocess_sentence(sentence, remove_stopwords = True):
    
    sentence = ' '.join(sentence)
    speech_pos = mecab.pos(sentence)
    clear_pos = [n for n, tag in speech_pos] #명사만
    sentence = ' '.join(clear_pos)
    
    # 불용어 제거 (Text)
    tokens = ' '.join(word for word in sentence.split())
    return tokens

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    with session.as_default():
        with session.graph.as_default():
            set_session(session)
            e_out, e_h, e_c = encoder_model.predict(input_seq)

     # <SOS>에 해당하는 토큰 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar_word_to_index['sostoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition: # stop_condition이 True가 될 때까지 루프 반복
        with session.as_default():
            with session.graph.as_default():
                set_session(session)
                output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tar_index_to_word[sampled_token_index]

        if(sampled_token!='eostoken'):
            decoded_sentence += ' '+sampled_token

        #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_token == 'eostoken'  or len(decoded_sentence.split()) >= (summary_max_len-1)):
            stop_condition = True

        # 길이가 1인 타겟 시퀀스를 업데이트
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # 상태를 업데이트 합니다.
        e_h, e_c = h, c

    return decoded_sentence

cat_list = [" 개인 및 관계", " 미용과 건강", " 상거래 쇼핑", " 시사교육", " 식음료", " 여가 생활", " 일과 직업", " 주거와 생활", " 행사"]

def pick_cat(input_text):
    fir_list=[]
    for i in range(len(input_text)):
        fir_list.append(decode_sequence(input_text[i].reshape(1, text_max_len)))
    
    cat_count=[]
    for i in range(len(cat_list)):
        cat_count.append(fir_list.count(cat_list[i]))
        
    max_count = max(cat_count)
    max_name = cat_list[cat_count.index(max(cat_count))]

    cat_count[cat_count.index(max(cat_count))] = 0

    max2_count = max(cat_count)
    max2_name = cat_list[cat_count.index(max(cat_count))]

    cat_count[cat_count.index(max(cat_count))] = 0
    
    if max_name == " 개인 및 관계":
        if max_count >= max2_count*1.5:
            cat_result = max_name
        else:
            cat_result = max2_name
    else:
        cat_result = max_name
    
    return cat_result


# 패딩길이
text_max_len = 75
summary_max_len = 4

set_session(session)

model = keras.models.load_model('model_cat2.h5', custom_objects={'AttentionLayer':AttentionLayer})

encoder_model = keras.models.load_model('model_en.h5', custom_objects={'AttentionLayer':AttentionLayer})
decoder_model = keras.models.load_model('model_de.h5', custom_objects={'AttentionLayer':AttentionLayer})

# loading
with open('tokenizer.pickle', 'rb') as s_handle:
    src_tokenizer = pickle.load(s_handle)
    
with open('tokenizer2.pickle', 'rb') as t_handle:
    tar_tokenizer = pickle.load(t_handle)
    
src_index_to_word = src_tokenizer.index_word # 원문 단어 집합에서 정수 -> 단어를 얻음
tar_word_to_index = tar_tokenizer.word_index # 요약 단어 집합에서 단어 -> 정수를 얻음
tar_index_to_word = tar_tokenizer.index_word # 요약 단어 집합에서 정수 -> 단어를 얻음
    
def main(intext):
    pp_text=[]
    fin_text=[]
    k=0

    target = intext.split(" ")   
    pp_text.append(preprocess_sentence(target))

    tkn_text = src_tokenizer.texts_to_sequences(pp_text)
    
    for i in range(len(tkn_text)):
        if not tkn_text[i-k]:
            del tkn_text[i-k]
            k+=1
        else:
            fin_text.append(np.array(tkn_text[i-k]))
            
    fin_text = pad_sequences(fin_text, maxlen = text_max_len, padding='post')

    result = pick_cat(fin_text)
    
    return result
