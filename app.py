import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import glob

# Membaca data
file_path = 'https://gist.githubusercontent.com/MeylaniEP/52022bf121263fe05921e6b114a23612/raw/e2a28786419c69783d177709134f5eb9cd489f9a/online_gamedev_behavior_dataset.csv'
data = pd.read_csv(file_path)

# Preprocessing
label_cols = ['EngagementLevel', 'Gender', 'GameGenre', 'GameDifficulty', 'Location']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Memilih fitur yang relevan
features = ['Age', 'Gender', 'GameGenre', 'GameDifficulty', 'AvgSessionDurationMinutes', 'Location']
X = data[features]

# Load Model
folder_path = 'dist'
pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))

if pkl_files:
    model_path = pkl_files[0]
else:
    print('File not found')

if model_path:
    model = joblib.load(model_path)
    
    # Aplikasi Streamlit
    st.title('Hallo Dev Game')
    location = st.selectbox('Location', options=['USA','Europe','Asia','Other'])
    age = st.slider('Age', min_value=15, max_value=50, value=20, step=1)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    game_genre = st.selectbox('Genre Game', options=['Strategy', 'Sports', 'Action', 'RPG', 'Simulation'])
    difficulty = st.selectbox('GameDifficulty', options=['Medium', 'Easy', 'Hard'])
    session_duration = st.slider('AvgSessionDurationMinutes', min_value=10, max_value=180, value=40, step=1)


    if st.button('Submit'):
        # Transformasi input user
        input_data = pd.DataFrame([[age, gender, game_genre, difficulty, session_duration, location]], columns=features)
        used_cols = [col for col in label_cols if col in features]
        for col in used_cols:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Mencari rekomendasi
        pred = model.predict(input_data)
        recommended_genres = label_encoders['EngagementLevel'].inverse_transform(pred)

        st.write('Keterlibatan Pemain :')
        for genre in recommended_genres:
            st.write(genre)

        # Penjelasan genre yang dipilih
        genre_descriptions = {
            'Strategy': 'Game strategi menekankan pemikiran strategis, perencanaan jangka panjang, dan pengelolaan sumber daya. Pemain biasanya mengatur unit, membangun struktur, dan mengelola ekonomi dalam permainan.',
            'Sports': 'Game olahraga mensimulasikan olahraga nyata atau fiktif. Pemain biasanya mengendalikan atlet atau tim untuk mencapai tujuan olahraga tertentu seperti mencetak gol atau memenangkan perlombaan.',
            'Action': 'Game aksi menekankan refleks dan keterampilan motorik. Pemain berinteraksi langsung dengan lingkungan permainan dengan mengalahkan musuh, menghindari rintangan, dan menyelesaikan misi.',
            'RPG': 'Game peran memungkinkan pemain mengambil peran karakter fiksi dan mengendalikannya. Terdapat unsur pengembangan karakter, seperti meningkatkan keterampilan atau level, serta narasi mendalam.',
            'Simulation': 'Game simulasi mensimulasikan aktivitas nyata atau skenario fiktif dengan tujuan memberikan pengalaman yang realistis kepada pemain. Biasanya melibatkan aspek kehidupan nyata seperti manajemen bisnis, kehidupan kota, atau simulasi pesawat.'
        }

        st.write(f'Penjelasan Genre {game_genre}:')
        st.write(genre_descriptions[game_genre])

        genre_examples = {
            'Strategy': [('Civilization', 'public/img/Civilization_VI.png'),
                         ('StarCraft', 'public/img/starcraft.jpg'),
                         ('Age of Empires', 'public/img/AgeOf.jpg')],
            'Sports': [('FIFA', 'public/img/FIFA.jpg'),
                       ('NBA 2K', 'public/img/NBA.jpg'),
                       ('Madden NFL', 'public/img/Madden.jpg')],
            'Action': [('Grand Theft Auto (GTA)', 'public/img/GTA5.jpg'),
                       ('Call of Duty', 'public/img/COD.jpg'),
                       ('Assassins Creed', 'public/img/AssasinCreed.jpg')],
            'RPG': [('The Elder Scrolls', 'public/img/TheElder.jpg'),
                    ('Final Fantasy', 'public/img/FinalFantasy.jpg'),
                    ('The Witcher', 'public/img/TheWitcher.jpg')],
            'Simulation': [('The Sims', 'public/img/TheSims.jpg'),
                           ('SimCity', 'public/img/simcity.jpg'),
                           ('Flight Simulator', 'public/img/flightSimulator.jpg')]
        }

        st.write(f'Contoh Game {game_genre}:')
        for example in genre_examples[game_genre]:
            st.image(example[1], caption=example[0], width=300)
else:
    st.write("Model not found")

