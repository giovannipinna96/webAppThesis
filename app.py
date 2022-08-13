import numpy as np
import streamlit as st
import torch
from PIL import Image

from net import create_models, predict_res_ii
from preprocessingImg import preprocessimg

st.markdown("""Punzone recognition""")
# load style.css
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.expander("Project introduction"):
    st.write("""
        In this section we will write about the goal of the project and how it was carried out in brief.
    """)
    st.write("""
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        magna aliqua. Tellus integer feugiat scelerisque varius morbi. Praesent semper feugiat nibh sed pulvinar proin
        gravida hendrerit lectus. Lacus viverra vitae congue eu consequat ac felis. Nibh sed pulvinar proin gravida
        hendrerit. Amet consectetur adipiscing elit pellentesque habitant morbi tristique. Ante metus dictum at
        tempor commodo. Libero nunc consequat interdum varius. Nisl rhoncus mattis rhoncus urna neque. Elementum
        nisi quis eleifend quam adipiscing vitae proin sagittis. Ullamcorper dignissim cras tincidunt lobortis
        feugiat vivamus at augue eget. Ante metus dictum at tempor commodo. Condimentum lacinia quis vel eros.
        Fermentum dui faucibus in ornare quam viverra orci sagittis eu. Eget dolor morbi non arcu. Id ornare
        arcu odio ut.

        A diam sollicitudin tempor id eu. Sit amet aliquam id diam. Donec et odio pellentesque diam.
        Sit amet cursus sit amet dictum sit amet justo. Eleifend donec pretium vulputate sapien nec sagittis aliquam.
        Eget nunc lobortis mattis aliquam faucibus purus in massa tempor. Libero nunc consequat interdum varius sit
        amet mattis. Id nibh tortor id aliquet. Pulvinar sapien et ligula ullamcorper malesuada proin libero nunc
        consequat. Commodo elit at imperdiet dui accumsan sit amet nulla.
    """)

with st.expander("How to use"):
    st.write("""
        Simple guide on how to use the web App and how to interpret the output. Also possible with photos or gifs.
    """)
    st.write("""
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        magna aliqua. Tellus integer feugiat scelerisque varius morbi. Praesent semper feugiat nibh sed pulvinar proin
        gravida hendrerit lectus. Lacus viverra vitae congue eu consequat ac felis. Nibh sed pulvinar proin gravida
        hendrerit. Amet consectetur adipiscing elit pellentesque habitant morbi tristique. Ante metus dictum at
        tempor commodo. Libero nunc consequat interdum varius. Nisl rhoncus mattis rhoncus urna neque. Elementum
        nisi quis eleifend quam adipiscing vitae proin sagittis. Ullamcorper dignissim cras tincidunt lobortis
        feugiat vivamus at augue eget. Ante metus dictum at tempor commodo. Condimentum lacinia quis vel eros.
        Fermentum dui faucibus in ornare quam viverra orci sagittis eu. Eget dolor morbi non arcu. Id ornare
        arcu odio ut.

        A diam sollicitudin tempor id eu. Sit amet aliquam id diam. Donec et odio pellentesque diam.
        Sit amet cursus sit amet dictum sit amet justo. Eleifend donec pretium vulputate sapien nec sagittis aliquam.
        Eget nunc lobortis mattis aliquam faucibus purus in massa tempor. Libero nunc consequat interdum varius sit
        amet mattis. Id nibh tortor id aliquet. Pulvinar sapien et ligula ullamcorper malesuada proin libero nunc
        consequat. Commodo elit at imperdiet dui accumsan sit amet nulla.
    """)

with st.expander("About the dataset"):
    st.write("""
        Briefly explain what a punch is, where they are found. It will also briefly describe how images are processed
        before giving them as input into the neural network for prediction.
    """)
    st.write("""
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        magna aliqua. Tellus integer feugiat scelerisque varius morbi. Praesent semper feugiat nibh sed pulvinar proin
        gravida hendrerit lectus. Lacus viverra vitae congue eu consequat ac felis. Nibh sed pulvinar proin gravida
        hendrerit. Amet consectetur adipiscing elit pellentesque habitant morbi tristique. Ante metus dictum at
        tempor commodo. Libero nunc consequat interdum varius. Nisl rhoncus mattis rhoncus urna neque. Elementum
        nisi quis eleifend quam adipiscing vitae proin sagittis. Ullamcorper dignissim cras tincidunt lobortis
        feugiat vivamus at augue eget. Ante metus dictum at tempor commodo. Condimentum lacinia quis vel eros.
        Fermentum dui faucibus in ornare quam viverra orci sagittis eu. Eget dolor morbi non arcu. Id ornare
        arcu odio ut.

        A diam sollicitudin tempor id eu. Sit amet aliquam id diam. Donec et odio pellentesque diam.
        Sit amet cursus sit amet dictum sit amet justo. Eleifend donec pretium vulputate sapien nec sagittis aliquam.
        Eget nunc lobortis mattis aliquam faucibus purus in massa tempor. Libero nunc consequat interdum varius sit
        amet mattis. Id nibh tortor id aliquet. Pulvinar sapien et ligula ullamcorper malesuada proin libero nunc
        consequat. Commodo elit at imperdiet dui accumsan sit amet nulla.
    """)

with st.expander("Prediction technique"):
    st.write("""
        A little more technical description of which network or networks are used for prediction.
        Brief summary of the results obtained from the tests and then treat the accuracy of the prediction.
        Brief description of the problems that may arise.
    """)
    st.write("""
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        magna aliqua. Tellus integer feugiat scelerisque varius morbi. Praesent semper feugiat nibh sed pulvinar proin
        gravida hendrerit lectus. Lacus viverra vitae congue eu consequat ac felis. Nibh sed pulvinar proin gravida
        hendrerit. Amet consectetur adipiscing elit pellentesque habitant morbi tristique. Ante metus dictum at
        tempor commodo. Libero nunc consequat interdum varius. Nisl rhoncus mattis rhoncus urna neque. Elementum
        nisi quis eleifend quam adipiscing vitae proin sagittis. Ullamcorper dignissim cras tincidunt lobortis
        feugiat vivamus at augue eget. Ante metus dictum at tempor commodo. Condimentum lacinia quis vel eros.
        Fermentum dui faucibus in ornare quam viverra orci sagittis eu. Eget dolor morbi non arcu. Id ornare
        arcu odio ut.

        A diam sollicitudin tempor id eu. Sit amet aliquam id diam. Donec et odio pellentesque diam.
        Sit amet cursus sit amet dictum sit amet justo. Eleifend donec pretium vulputate sapien nec sagittis aliquam.
        Eget nunc lobortis mattis aliquam faucibus purus in massa tempor. Libero nunc consequat interdum varius sit
        amet mattis. Id nibh tortor id aliquet. Pulvinar sapien et ligula ullamcorper malesuada proin libero nunc
        consequat. Commodo elit at imperdiet dui accumsan sit amet nulla.
    """)

img = st.file_uploader("Update Image")
show_file = st.empty()

if not img:
    show_file.info("Please upadate a file")
else:
    st.image(Image.open(img), use_column_width=True)

    # code for create the net and predict the image
    input_model = preprocessimg(img)
    res, res_triplet, clf, sc, res_ii, threshold, mean = create_models(17)
    res.eval()
    #res_triplet.eval()
    res_ii.eval()

    pred_res = res(input_model)
    prob_res = torch.nn.functional.softmax(pred_res, dim=1).detach().numpy()

    #pred_res_triplet = res_triplet(input_model).detach().numpy()
    #prob_res_triplet = clf.predict_proba(sc.transform(pred_res_triplet))

    pred_res_ii, out_y = predict_res_ii(res_ii, threshold, mean, input_model)
    prob_res_ii = torch.nn.functional.softmax(out_y, dim=1).numpy()
    st.write(clf.best_params_)

    a1, a3 = st.columns(2)
    a1.metric("ResNet50: class", f'{np.argmax(prob_res)}')
    #a2.metric("ResNet50 triplet: class", f'{np.argmax(prob_res_triplet, axis=1)}')
    if pred_res_ii == -1 or prob_res_ii.max() < 0.21:
        a3.metric("ResNet50 ii-loss: class", 'Unknown')
    else:
        a3.metric("ResNet50 ii-loss: class", f'{pred_res_ii}')
    b1, b3 = st.columns(2)
    b1.metric("ResNet50: accurancy", f'{prob_res.max() * 100:3.2f} %')
    #b2.metric("ResNet50 triplet: accurancy", f'{prob_res_triplet.max() * 100:3.2f} %')
    if pred_res_ii == -1 or prob_res_ii.max() < 0.21:
        b3.metric("ResNet50 ii-loss: accurancy", 'Unknown')
    else:
        b3.metric("ResNet50 ii-loss: accurancy", f'{prob_res_ii.max() * 100:3.2f} %')
