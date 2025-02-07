# Siamese Face Verification System
This Face Verification System based on Siamese Networks utilizes Triplet Loss to determine the similarity between two face images.
The model is trained to identify whether two faces belong to the same person or not by comparing their embeddings.

![](https://lh6.googleusercontent.com/FURxmOgv589pwu8ZnMhIyxdznRXV7VfIAjtdxSSxNCAYmdfAlAwnSLFiQ8YtEv_l-3srghGaQaB-DVT72dOKikzqEYffPG9QHAseryruCfLg01CnQiltx1dXM7n0Y5o7uHSk99AsD0r9wzZO5dd7Iw)


## Siamese model and triplet loss
I recommend Andrew NG's video on [Siamese models](https://www.youtube.com/watch?v=6jfw8MuKwpI) and [triplet loss](https://www.youtube.com/watch?v=d2XB5-tuCWU) to better understand the concepts.


## Dataset
[The Pins face recognition dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) has hundreds of images for each celebrity, and it is quite useful for this task.


## User interface
![image](https://github.com/user-attachments/assets/d8b09026-27df-4e16-8771-80a72ef3987e)


## requiremnets
```
pip install streamlit tensorflow opencv-python numpy scikit-learn fastapi uvicorn
```
fastapi is used in the backend and streamlit works as user interface


## How to use 
```
git clone https://github.com/RozhanMk/Face-Verification-System
cd Face-Verification-System
cd backend
uvicorn main:app --reload
```
in another terminal run this:
```
cd frontend
streamlit run app.py 
```
Now you can register faces and verify them!


## Resources
https://www.kaggle.com/code/stoicstatic/face-recognition-siamese-w-triplet-loss

https://github.com/Mitix-EPI/Face-Recognition-using-Siamese-Networks



