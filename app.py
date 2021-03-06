import streamlit as st
from PIL import Image

from net import create_model, predictimg
from preprocessingImg import preprocessimg

st.markdown("<h1 style='text-align: center;'>Punzone recognition</h1>", unsafe_allow_html=True)
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
    model = create_model()
    results = predictimg(model, input_model)

    # for i in range(len(results)):
    #     st.text(
    #         f"Net prediction {i} , label : {' '.join(results[i][0].split()[1:])} \n\t\t probability : {results[i][1]}%")

    a1, a2, a3 = st.columns(3)
    a1.metric("ResNet50", f'{round(results[0][1], 2)} %')
    a2.metric("Method2", f'{round(results[1][1], 2)} %')
    a3.metric("Method3", f'{round(results[2][1], 2)} %')
