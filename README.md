# Player Recognition System

This project is a face recognition system specifically designed for identifying professional soccer players. It combines a pre-trained FaceNet model and a custom KNN classifier for embedding classification. The application allows users to upload an image of a soccer player and identifies the player from a pre-defined dataset.

## Features

- **Image Downloading**: Automates the collection of player images from Bing using the `bing_image_downloader` library.
- **Image Preprocessing**: Detects, crops, and resizes facial regions using MTCNN.
- **Face Embedding Extraction**: Generates embeddings using the FaceNet model for face representation.
- **Player Recognition**: Classifies players using a KNN model trained on the generated embeddings.
- **Web Interface**: A Flask-based web application for uploading and analyzing images.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MaikolCid/IA-Football-Players-Recognition.git
   cd IA-Football-Players-Recognition-main

![image](https://github.com/user-attachments/assets/03add1b9-a9bb-489f-a9f9-cb614654b082)
![image](https://github.com/user-attachments/assets/514220a9-37d0-4550-ab02-dd457d0c1878)
