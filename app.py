import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np




# Load emotion detection model
classifier = load_model('facialemotionmodel.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

acousticness = 0.5
danceability = 0.5
energy = 0.5
instrumentalness = 0.5
valence = 0.5
tempo = 120.0

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_emotion = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            output = "Unknown"
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.reshape(roi, (1, 48, 48, 1))
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                self.latest_emotion = emotion_labels[maxindex]
                output = self.latest_emotion
                with open('latest_emotion.txt', 'w') as file:
                    file.write(self.latest_emotion)
            label_position = (x, y - 10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
    
    def get_latest_emotion(self):
        return self.latest_emotion

video_transformer = VideoTransformer()

# Load song recommendation data
def load_data():
    df = pd.read_csv("filtered_track_df.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df
exploded_track_df = load_data()
genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]


genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]



emotion_presets = {
    'Angry': {'acousticness': 0.2, 'danceability': 0.6, 'energy': 0.8, 'instrumentalness': 0.1, 'valence': 0.3, 'tempo': 140.0},
    'Disgust': {'acousticness': 0.5, 'danceability': 0.3, 'energy': 0.4, 'instrumentalness': 0.2, 'valence': 0.2, 'tempo': 100.0},
    'Fear': {'acousticness': 0.7, 'danceability': 0.3, 'energy': 0.2, 'instrumentalness': 0.5, 'valence': 0.1, 'tempo': 90.0},
    'Happy': {'acousticness': 0.4, 'danceability': 0.8, 'energy': 0.7, 'instrumentalness': 0.1, 'valence': 0.9, 'tempo': 120.0},
    'Sad': {'acousticness': 0.8, 'danceability': 0.2, 'energy': 0.2, 'instrumentalness': 0.3, 'valence': 0.1, 'tempo': 80.0},
    'Surprise': {'acousticness': 0.4, 'danceability': 0.7, 'energy': 0.6, 'instrumentalness': 0.2, 'valence': 0.7, 'tempo': 130.0},
    'Neutral': {'acousticness': 0.5, 'danceability': 0.5, 'energy': 0.5, 'instrumentalness': 0.2, 'valence': 0.5, 'tempo': 110.0},
}

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

import os

if os.path.exists('latest_emotion.txt'):
    with open('latest_emotion.txt', 'r') as file:
        detected_emotion = file.read().strip()
        if detected_emotion in emotion_presets:
            selected_emotion = detected_emotion

# Get the preset values for the sliders based on the selected emotion
if selected_emotion and selected_emotion in emotion_presets:
    acousticness_default = emotion_presets[selected_emotion]['acousticness']
    danceability_default = emotion_presets[selected_emotion]['danceability']
    energy_default = emotion_presets[selected_emotion]['energy']
    instrumentalness_default = emotion_presets[selected_emotion]['instrumentalness']
    valence_default = emotion_presets[selected_emotion]['valence']
    tempo_default = emotion_presets[selected_emotion]['tempo']
else:
    # Default values if no emotion detected
    acousticness_default = 0.5
    danceability_default = 0.5
    energy_default = 0.5
    instrumentalness_default = 0.5
    valence_default = 0.5
    tempo_default = 120.0

title = "Spotify Song Recommendation Engine"
st.title(title)

st.write("First of all, welcome! This is the place where you can customize what you want to listen to based on genre and several key audio features. Play around with different settings and listen to the songs recommended by our system!")
st.markdown("##")

with st.container():
    col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio("", genre_names, index=genre_names.index("Pop"))
    with col4:
        st.markdown("***Choose an emotion:***")
        selected_emotion = st.selectbox('', list(emotion_presets.keys()) + ["Start Camera"], index=list(emotion_presets.keys()).index('Neutral'))
        
        if selected_emotion == "Start Camera":
            st.write("Starting camera for emotion detection...")
            webrtc_streamer(key="example", video_processor_factory=lambda: video_transformer)
            if st.button('Stop'):
                try:
                    with open('latest_emotion.txt', 'r') as file:
                        latest_emotion = file.read()
                    st.write(f'Latest detected emotion: {latest_emotion}')
                except FileNotFoundError:
                    st.write("No emotion detected yet.")
        else:
            acousticness = emotion_presets[selected_emotion]['acousticness']
            danceability = emotion_presets[selected_emotion]['danceability']
            energy = emotion_presets[selected_emotion]['energy']
            instrumentalness = emotion_presets[selected_emotion]['instrumentalness']
            valence = emotion_presets[selected_emotion]['valence']
            tempo = emotion_presets[selected_emotion]['tempo']

    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider('Select the year range', 1990, 2019, (2015, 2019))
        acousticness = st.slider('Acousticness', 0.0, 1.0, acousticness_default)
        danceability = st.slider('Danceability', 0.0, 1.0, danceability_default)
        energy = st.slider('Energy', 0.0, 1.0, energy_default)
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, instrumentalness_default)
        valence = st.slider('Valence', 0.0, 1.0, valence_default)
        tempo = st.slider('Tempo', 0.0, 244.0, tempo_default)


test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
tracks_per_page = 6
tracks = []
for uri in uris:
    track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
    tracks.append(track)

if 'previous_inputs' not in st.session_state:
    st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat

current_inputs = [genre, start_year, end_year] + test_feat
if current_inputs != st.session_state['previous_inputs']:
    if 'start_track_i' in st.session_state:
        st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

if 'start_track_i' not in st.session_state:
    st.session_state['start_track_i'] = 0



with st.container():
    col1, col2, col3 = st.columns([2,1,2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page
    current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i%2==0:
                with col1:
                    components.html(track, height=400)
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(r=audio[:5], theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)
            else:
                with col3:
                    components.html(track, height=400)
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(r=audio[:5], theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)
    else:
        st.write("No songs left to recommend")
