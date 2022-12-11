import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.cache(allow_output_mutation=True)
def load_models():
    print('chargement model')
    model_eval = tf.keras.models.load_model(
        "models/increase_vocab-3/model.h5", compile=True)
    # model_auto = load_model(‘models/auto_model.h5’, compile=False)
    with open('models/increase_vocab-3/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return model_eval, tokenizer


result = 0



def predict():
    print(txt)
    model, tokenizer = load_models()

    text_preprocessing = tokenizer.texts_to_sequences([txt])
    text_preprocessing_pad = pad_sequences(text_preprocessing, maxlen=200)
    # text_preprocessing_pad
    return round(model.predict(text_preprocessing_pad)[0][0].item(), 2)


col1, col2 = st.columns([3, 1])

with col1:
    txt = st.text_area('Text to analyze', '', placeholder='''
        Enter a text ...
        ''')
    button = st.button('Read')

with col2:
    info = st.empty()
    metric = st.empty()

    slider = st.empty()


if button:
    result = predict()
    print("result=", result)
    metric.metric(label="Indecide readibilty", value=result,
                  delta_color="inverse",
                  )

    slider.slider(
        'Readability degree',
        value=result,
        max_value=4.,
        min_value=-4.,
        disabled=True
    )
    if(result < -2.):
        info.success('✅  easy')
    elif(result > -2. and result < -1.):
        info.info('ℹ️    Pretty easy')
    elif(result > -1. and result < 1.):
        info.warning('ℹ️  Neutral')
    elif(result > 1. and result < 2.):
        info.warning('⚠️  Pretty hard')
    elif(result > 2.):
        info.error('🚨  Hard')
