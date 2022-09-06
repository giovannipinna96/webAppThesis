import numpy as np
import streamlit as st
import torch
from PIL import Image

from net import create_models, predict_res_ii
from preprocessingImg import preprocessimg

st.markdown("<h1 style='text-align: center;'>Punches recognition</h1>", unsafe_allow_html=True)
# load style.css
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.expander("Project introduction"):
    st.write("""
        The goal of this thesis is to be able to develop a computer vision system specialized in being able to catalog
        the image of a punch in the correct category. This project belongs to the field of digital humanities that
        emerged in America in the 1980s, but only in recent years has it expanded. The main problem in this field is
        that still much of the documents, paintings and art objects have not yet been digitized and enriched by the
        correct metadata so it is not always easy to find sufficiently rich and descriptive datasets to train
        artificial intelligence models correctly. 

        This work aims to investigate and help art historians to automatically identify and classify particular art
        forms, called punches, from Florentine paintings dated around 1400. This work was possible because, following
        the 1966 Florence flood there was a huge amount of work restoring and cataloguing the various works, and most
        recently the digitization of that archive, which allowed the creation of part of the dataset used.

        Classifying the punches in automatic way, avoiding to make all the measurements by hand, can be an important
        advantage for an art historian. Our proposal is to operate in another way, so no longer collecting the images in
        contact with the art, but working with high resolution images. These images will then be accompanied by a
        3D model to ensure that the photograph correctly reproduces the ratios. The results obtained will be elaborated 
        so as to be able to fully exploit the application advantage of artificial intelligence. With these prerogatives,
        this tool could be important for researchers in this field because the comparative investigation on punches is 
        a useful tool to verify the attributions of the works of art. If there are sufficiently specific data, an 
        interpretation of the occurrence of a punch can also help to establish the chronology of the works. 
        Until now only a small amount of these artworks are incontestably attributed, with certainty, to a name and 
        a precise date.

        In this work we go on to investigate the performance of "classical" techniques, which often are based on 
        key-points search. We also analyzes the most modern deep learning techniques. The deep learning models 
        underlying all the techniques is the ResNet50 which, due to its revolutionary architecture, has allowed us to 
        reach unprecedented values of accuracy in image classification.
    """)

with st.expander("How to use"):
    st.write("""
        This webapp is really easy to use! First you need to make sure that you have an internet connection and are 
        connected with a device with a camera to take a picture of the punch to be analyzed or with a device that 
        already has images of punches saved on it.

        At this point you will only need to click on the "Browse file" button and choose to upload the photo or take a 
        picture of the punch. It must be remembered that for the classification to be successful, it is necessary that 
        only one punch is present in the image provided to the webapp and that it is framed as best as possible. 

        Once this is done you will need to wait a few seconds is the probability of belonging to a class will be shown 
        in the gray boxes at the end of the page (below the photo).
    """)

with st.expander("About the dataset"):
    st.write("""
        The original dataset that we have consists in 3103 images of punches of which 2476 belong to the train set, 
        while the remaining 627 to the test set. In the train set the classes range from a maximum of elements 
        (class 729) of 490 images to a minimum of 13 images (class 438), therefore with an average of elements per 
        class equal to 145. For the test set the classes range from a maximum of 123 elements 
        (class 729) to a minimum of elements equal to 4 (class 438) with an average of images per class equal to 36.

        The classes of punches taken into consideration and also present in the Skahug archive  are:
        43, 72, 96, 171, 178, 274, 296, 371, 385, 385_FORSE, 391-431 (is a single label), 438, 619, 655, 657, 659, 729
        """)
    st.image(Image.open("all_puches.png"), use_column_width=True)
    st.write("""
        To better classify the punches, it was necessary to artificially create images (data augmentation). To do this, 
        we performed transformations to the images of the original dataset in order to create new ones, but similar to 
        the punches we already possessed. The transformations we applied are: 

            1. resize the image in 256x256 
            2. random rotation (0,359) with bilinear interpolation 
            3. random gaussian blur (kernel size = 5, probability = 0.3) 
            4. transform to tensor 
            5. random patch 
            6. tensor normalization
    """)

with st.expander("Prediction technique"):
    st.write("""
        In this paper we started immediately with fairly complex deep learning techniques. The first network we 
        implemented was ResNet50 pre-trained on ImageNet weights. In this way we could immediately understand how high
        the accuracy was achieved and what performance we could expect from similar networks.

        In the training phase the data were preprocessed, we used the optimizer RAdam with a learning rate 
        of 0.001.  Regarding loss initially we used cross entropy loss, but for more advanced techniques we it was
        changed. Finally, as a sheduler, which is an optimization algorithm, we chose the MultiStepLR with milestones
        of 5 and 15 and the gamma parameter of 0.1. More information about the scheduler can be found in the official
        Pythorch documentation. All algorithms were allied for 20 epochs.
        
        Specifically for classification we modified the FCN of the ResNet. In fact ResNet architecture presents as 
        the last layer a fully connected of 1000 classes. This because it was designed for the 1000-class ImageNet 
        classification. This configuration for us did not fit having only 18 classes. So as is usual to do in such 
        cases we replaced that layer with another fully connected one, but one that would output only 18 classes.

        The other classification technique is the ResNet50 with the ii-loss.  In this case, the network is trained 
        with both ii-loss and cross entropy loss. The cross entropy loss is used to train the classifier placed as 
        the output of the network that is responsible for assigning the correct output lable to an input. While the 
        ii-loss is used by the network to distance in the features space, which we place at 32 dimensions, the inputs. 
        In fact, thanks to a threshold found empirically we are able to identify whether an input belongs to one of the 
        training classes. If that input belongs to the training classes then it will pass through the classifier and be 
        classified, otherwise it will be classified as "unknown." 
        
        This technique has reported encouraging results although slightly below the accuracies achieved using ResNet50 
        alone. But this is understandable because it manages a more complex problem and perhaps with more data it would 
        have been able to achieve higher levels of accuracy. The accuracy of this network is around 96% for 
        classification and comes close to 70% in the task of determining whether or not an image belongs to the classes 
        known by the network.
    """)

img = st.file_uploader("Update Image")
show_file = st.empty()

if not img:
    show_file.info("Please upadate a file")
else:
    st.image(Image.open(img), use_column_width=True)

    # create dict for output lables
    lables = {'0': '171', '1': '178_FORSE', '2': '274', '3': '296', '4': '371', '5': '385', '6': '385_FORSE', '7': '391-431', '8': '43', '9': '438',
              '10': '619', '11': '655', '12': '657', '13': '659', '14': '72', '15': '729', '16': '96'}
    # code for create the net and predict the image
    input_model = preprocessimg(img)
    res, res_ii, threshold, threshold2, mean = create_models(17)
    res.eval()
    res_ii.eval()
    pred_res = res(input_model)
    prob_res = torch.nn.functional.softmax(pred_res, dim=1).detach().numpy()

    out_z, outlier_score_val, pred_res_ii, out_y = predict_res_ii(res_ii, threshold, threshold2, mean, input_model)
    prob_res_ii = torch.nn.functional.softmax(out_y, dim=1).numpy()

    # output ResNet50
    a1, a2 = st.columns(2)
    a1.metric("ResNet50: class", f'{lables.get(str(np.argmax(prob_res)))}')
    if pred_res_ii == -1:
        a2.metric("ResNet50 ii-loss: class", 'Unknown')
    else:
        a2.metric("ResNet50 ii-loss: class", f'{lables.get(str(pred_res_ii))}')

    # output ResNet50 + ii-loss
    b1, b2 = st.columns(2)
    b1.metric("ResNet50: confidence", f'{prob_res.max() * 100:3.2f} %')
    if pred_res_ii == -1:
        b2.metric("ResNet50 ii-loss: confidence", 'Unknown')
    else:
        b2.metric("ResNet50 ii-loss: confidence", f'{prob_res_ii.max() * 100:3.2f} %')
