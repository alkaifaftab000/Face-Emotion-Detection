# 😊 Face Emotion Detection Project

This project aims to detect emotions (happy or sad) from images using deep learning techniques. It's perfect for exploring the intersection of technology and human emotions.

---

## 🎯 Workflow

1. **Data Collection**:  
   - Downloaded 1,000 images featuring sad and happy faces, including babies, children, adults, and elderly people.  
   - Detect faces directly using a camera or by adding your own images.

2. **Data Preprocessing**:  
   - Removed irrelevant images (like dog pictures 🐕) using `cv2` and `imghdr`.  
   - Normalized pixel values for consistency in processing.

3. **Building the Model**:  
   - Employed **Keras** with **TensorFlow** backend to build a convolutional neural network (CNN):  
      - **Layers**:  
        - Conv2D  
        - MaxPooling2D  
        - Dense  
        - Flatten  
      - Small filters in Conv2D to extract intricate features.  
   - The model outputs a score:  
      - Close to `0`: Person is sad 😢  
      - Close to `1`: Person is happy 😃

---

## 🚀 Learning Outcomes

1. Understanding the preprocessing techniques to handle image-based data effectively.  
2. Building a deep learning model with CNNs to extract features and classify emotions.

---
## UI 
![{C82E27E4-26F4-4A42-BAE1-F45701B187C0}](https://github.com/user-attachments/assets/91975f13-2ceb-4b66-ac0d-2e22e4e404f9)
![{6C127782-44F0-4B2E-965B-620450302770}](https://github.com/user-attachments/assets/fc4ff745-6dc2-4114-a212-27cfe79360d8)

## Result
![{F9B3092B-A6E7-4319-9D27-5071B30384E7}](https://github.com/user-attachments/assets/7531621b-b997-4701-8396-2c3a7aba85f1)
---

## 🔧 Plugins Used

- `flask==3.0.2`  
- `tensorflow==2.18.0`  
- `pillow==10.2.0`  
- `numpy==1.26.4`  
- `gunicorn==21.2.0`  

---

