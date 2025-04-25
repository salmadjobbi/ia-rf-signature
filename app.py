

import streamlit as st
import joblib
import numpy as np
import cv2
from skimage.feature import hog
from PIL import Image
from io import BytesIO


print("skimage.feature is working!")
import sklearn
print("Scikit-learn version:", sklearn.__version__)

# Charger le modèle et le scaler
# Charger le modèle
model = joblib.load("C:\\Users\\salma\\Downloads\\rf_signature_model.sav")

# Charger le scaler
scaler = joblib.load("C:\\Users\\salma\\Downloads\\scaler_signature_model (1).sav")
print("Bonjour Salma, ton app fonctionne !")

# Fonction d'extraction des caractéristiques (HOG + AIS)
def extract_ais(img, grid_size=(6, 6)):
    h, w = img.shape
    block_h, block_w = h // grid_size[0], w // grid_size[1]
    features = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            block = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            features.append(np.mean(block))
    return features

def extract_features(images):
    ais_features = []
    hog_features = []
    
    for img in images:
        # Preprocess - normalisation et suppression du bruit
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Extraction des caractéristiques AIS (36 caractéristiques pour une grille 6x6)
        ais_features.append(extract_ais(img))
        
        # Extraction des caractéristiques HOG
        hog_feat = hog(img, orientations=8, pixels_per_cell=(16, 16), 
                      cells_per_block=(1, 1))
        hog_features.append(hog_feat)
    
    return np.concatenate((np.array(ais_features), np.array(hog_features)), axis=1)

# Fonction de prédiction
def predict_signature(image):
    # Convertir l'image en niveaux de gris
    img = np.array(image.convert('L'))  # Convertir en niveau de gris
    img = cv2.resize(img, (200, 200))  # Redimensionner l'image
    
    # Extraire les caractéristiques
    features = extract_features([img])
    
    # Normaliser les caractéristiques
    features_scaled = scaler.transform(features)
    
    # Prédiction avec le modèle
    prediction = model.predict(features_scaled)
    return prediction[0]

# Fonction principale pour l'interface Streamlit
def main():
    st.title("Prédiction de la Signature")
    
    uploaded_file = st.file_uploader("Téléchargez une image de signature", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Afficher l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image de signature", use_column_width=True)
        
        # Effectuer la prédiction
        if st.button("Prédire la signature"):
            prediction = predict_signature(image)
            
            if prediction == 0:
                st.success("La signature est authentique.")
            else:
                st.error("La signature est falsifiée.")
    
if __name__ == "__main__":
    main()
