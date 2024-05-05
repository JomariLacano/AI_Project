import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

class_names = ['cat', 'dog', 'panda']

model = load_model('animal_classification_model.h5')

def process_frame():
    global cap
    global video_label

    ret, frame = cap.read()
    if ret:
        try:
            resized_frame = cv2.resize(frame, (600, 500))
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            predicted_class = np.argmax(prediction)
            class_name = class_names[predicted_class]
            confidence_score = prediction[0][predicted_class]
            confidence_percentage = min(round(confidence_score * 100, 2), 99.99)
            label.config(text=f"What animal: {class_name} ({confidence_percentage}%)")

            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            video_label.configure(image=img)
            video_label.image = img
        except Exception as e:
            print(f"An error occurred: {e}")
            on_closing()  # Call on_closing to release the camera and close the window
    video_label.after(10, process_frame)

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (55, 32))  # Resize to match the expected input shape of the model
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    processed_frame = rgb_frame.astype('float32') / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame


def on_closing():
    global cap
    cap.release()
    root.destroy()

try:
    root = tk.Tk()
    root.title("Cat Dog Panda Identification")
    root.geometry("700x600")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    video_label = tk.Label(root)
    video_label.pack()

    label = tk.Label(root, text="", font=("Helvetica", 16))
    label.pack(pady=10)

    cap = cv2.VideoCapture(0)

    process_frame()

    root.mainloop()

except Exception as e:
    print(f"An error occurred: {e}")
    on_closing()
