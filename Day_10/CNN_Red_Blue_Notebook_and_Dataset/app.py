import cv2
import numpy as np
import tensorflow as tf

# 1. Load the saved CNN model
loaded_model = tf.keras.models.load_model("red_blue_cnn_model.keras")
class_names = ["blue", "red"]  # 0: blue, 1: red

# 2. Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Preprocess camera frame (BGR -> RGB, resize to 64x64)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_batch = np.expand_dims(img_resized.astype("float32"), axis=0)

    # 4. Predict class
    prob = loaded_model.predict(img_batch, verbose=0)[0][0]
    pred_label = class_names[1] if prob >= 0.5 else class_names[0]
    confidence = prob if prob >= 0.5 else (1.0 - prob)

    # 5. Draw prediction on the live video frame
    text = f"Pred: {pred_label} ({confidence * 100:.1f}%)"
    color = (0, 0, 255) if pred_label == "red" else (255, 0, 0)  # BGR color
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("CNN Camera Prediction", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()