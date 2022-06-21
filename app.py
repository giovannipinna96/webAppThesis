import streamlit as st

st.markdown("<h1 style='text-align: center; color: red;'>Punzone recognition</h1>", unsafe_allow_html=True)

with st.expander("Project introduction"):
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
    st.image(img, use_column_width=True)
    # code for create the net and predict the image
