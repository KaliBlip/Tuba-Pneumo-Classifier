
# test_model_gui.py

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model("TuberPneu_model.h5")

# GUI setup
root = tk.Tk()
root.title("Chest X-ray Classifier")
root.geometry("600x600")

# Widgets
img_label = Label(root)
img_label.pack()

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict(path):
    img_array = preprocess_image(path)
    prediction = model.predict(img_array)[0][0]
    label = "Tuberculosis" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    if label == "Tuberculosis":
        if confidence > 0.90:
            stage = "Advanced stage"
        elif confidence > 0.70:
            stage = "Intermediate stage"
        else:
            stage = "Early stage"
    else:
        stage = "N/A"

    result_label.config(text=f"Prediction: {label}\nConfidence: {confidence*100:.2f}%\nStage: {stage}")

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        img_label.config(image=photo)
        img_label.image = photo
        predict(file_path)

upload_btn = Button(root, text="Upload Chest X-ray", command=upload_image)
upload_btn.pack(pady=20)

root.mainloop()
