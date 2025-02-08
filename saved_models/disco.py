import h5py

file_path = "C:/Users/Shabiul/Downloads/multiple-disease-prediction-streamlit-app-main/saved_models/brain.h5"

with h5py.File(file_path, "r") as f:
    print(list(f.keys()))  # Check the contents of the .h5 file
