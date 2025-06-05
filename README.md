🌿 Plant Disease Detection

A deep learning project to detect plant diseases from images using computer vision.

⸻

📁 Project Structure
	•	Client:
	•	Mobile App: Built with Flutter
	•	Web Interface: HTML/CSS/JavaScript, served via Flask
	•	Server:
	•	Flask Backend: Hosts the trained deep learning model and exposes REST APIs

⸻

📌 Project Overview

This system utilizes Detectron2 (Mask R-CNN R50-FPN 3x) for accurate plant disease detection. The model is deployed on a Flask server, accessible through:
	•	A Flutter-based mobile app
	•	A web client interface

🔍 Model detects:
	•	3 Potato diseases
	•	4 Tomato diseases

⸻

🛠️ Tech Stack

Frontend
	•	🌐 Web: HTML / CSS / JavaScript (served via Flask)
	•	📱 Mobile: Flutter

Backend
	•	🚀 Flask: RESTful API server
	•	🧠 Detectron2: Deep learning framework (Mask R-CNN)
	•	🌍 Ngrok: For public URL tunneling during development/testing

Dataset & Annotation
	•	📂 Source: Kaggle
	•	🏷 Annotation Tool: Makesense.AI