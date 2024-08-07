
# PDF Reader with FastAPI

This project is a PDF reader and summarizer application built using FastAPI. It allows users to upload PDF files, extract and optionally summarize the text, and convert the text to speech. The application uses modern machine learning models for text summarization and text-to-speech conversion, and provides real-time progress updates via WebSockets.

## Features

- **PDF Upload and Text Extraction**: Upload PDF files and extract text using PyPDF2.
- **Text Summarization**: Optional summarization of the extracted text using the T5 model from Hugging Face's Transformers library.
- **Text-to-Speech Conversion**: Convert extracted (or summarized) text to speech using gTTS and Pydub.
- **Real-time Progress Updates**: Receive real-time updates on the processing status via WebSocket.
- **Auto Scroll**: Optionally highlight and auto-scroll text segments in sync with the audio playback.

## Technologies Used

- **FastAPI**: For building the web application and handling requests.
- **WebSockets**: For real-time progress updates.
- **PyPDF2**: For extracting text from PDF files.
- **Hugging Face Transformers**: For text summarization using the T5 model.
- **gTTS and Pydub**: For text-to-speech conversion.
- **Jinja2**: For rendering HTML templates.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/sirajjunior540/PDFReader.git
    cd PDFReader
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```

5. **Open your browser and navigate to** `http://localhost:8000` **to access the application**.

## Project Structure

- **app.py**: The main application file containing all endpoints and logic.
- **templates/**: Directory containing HTML templates.
- **static/**: Directory containing static files like CSS and JavaScript.
- **uploads/**: Directory where uploaded PDF files are stored.
- **audio_segments/**: Directory where generated audio segments are stored.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Developed by Abdallah Ahmed.
