import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mnist_model.h5')

def process_frame():
    global cap
    global video_label

    ret, frame = cap.read()
    if ret:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                digit = thresh[y:y+h, x:x+w]
                resized_digit = cv2.resize(digit, (28, 28))
                resized_digit = resized_digit.astype('float32') / 255.0
                resized_digit = np.expand_dims(resized_digit, axis=-1)
                resized_digit = np.expand_dims(resized_digit, axis=0)
                prediction = model.predict(resized_digit)
                digit_class = np.argmax(prediction)
                confidence = min(np.max(prediction) * 100, 99.99)
                label.config(text=f"Predicted Digit: {digit_class}\nConfidence: {confidence:.2f}%")
            else:
                label.config(text="No digit detected")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            video_label.configure(image=img)
            video_label.image = img
        except Exception as e:
            print(f"An error occurred: {e}")
            on_closing()  # Call on_closing to release the camera and close the window
    video_label.after(10, process_frame)

def on_closing():
    global cap
    cap.release()
    root.destroy()

try:
    root = tk.Tk()
    root.title("MNIST Digit Recognition")
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
