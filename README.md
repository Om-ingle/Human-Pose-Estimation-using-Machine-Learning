# Human-Pose-Estimation-using-Machine-Learning


An interactive web application for estimating human poses in images using OpenCV's deep learning module and TensorFlow.

---

## Features

- **Upload Image**: Allows users to upload their own image for pose estimation.
- **Default Image Support**: Uses a preloaded demo image if no file is uploaded.
- **Customizable Threshold**: Set the detection confidence threshold using a slider.
- **Real-time Pose Estimation**: Detects key body joints and visualizes connections between them.
- **Interactive UI**: Built with Streamlit for an intuitive and visually appealing interface.

---

## Demo

![Demo GIF](https://example.com/demo-gif) *(Replace with actual link or image)*

---

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- A compatible deep learning model file (`graph_opt.pb`)

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Human-Pose-Estimation-using-Machine-Learning.git
    cd Human-Pose-Estimation-using-Machine-Learning
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

4. Open the app in your browser at [http://localhost:8501](http://localhost:8501).

---

## File Structure

```
Human-Pose-Estimation-using-Machine-Learning/
├── .venv/                          # Virtual environment
├── venv/                           # Another virtual environment
├── .gitattributes                  # Git attributes file
├── OutPut-image.png                # Example output image
├── README.md                       # Project README file
├── estimation_app1.py              # Estimation application script
├── graph_opt.pb                    # Pre-trained TensorFlow model
├── output.mov                      # Example output video
├── pose-gif.gif                    # Pose estimation demo GIF
├── pose_estimation.py              # Script for image-based pose estimation
├── pose_estimation_Video.py        # Script for video-based pose estimation
├── requirements.txt                # Python dependencies
├── run.jpg                         # Example input image (run pose)
├── run.mov                         # Example input video (run pose)
├── run1.mp4                        # Another example input video
└── stand.jpg                       # Default demo image

```

---

## Usage

1. **Upload an Image**: Use the sidebar to upload a JPG, JPEG, or PNG image.
2. **Set Detection Threshold**: Adjust the slider to change the confidence level for pose detection.
3. **View Results**: Observe the estimated pose displayed below the uploaded image.

---

## Technologies Used

- **Streamlit**: For creating the web interface.
- **OpenCV**: For image processing and pose estimation.
- **TensorFlow**: For leveraging the pre-trained neural network model.

---

## Contributing

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature-name'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- OpenCV's Human Pose Estimation module.
- Streamlit for an excellent framework for building data applications.

---

## Author

**OM**  
Feel free to reach out for collaboration or queries!
