import os
import time
import PyPDF2
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gtts import gTTS
from pydub import AudioSegment

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clients = []

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
TEMPLATES = Jinja2Templates(directory="templates")

# Initialize T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

MAX_TOKEN_LENGTH = 512

# Function to generate a summary using T5 model
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids[:, :MAX_TOKEN_LENGTH]  # Ensure input length does not exceed max length
    outputs = model.generate(input_ids, max_length=MAX_TOKEN_LENGTH, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1  # +1 for the space
        if current_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    chunks.append(" ".join(current_chunk))  # add the last chunk
    return chunks

# Extract text from PDF using PyPDF2
def extract_text_from_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return pdf_text

# Text-to-speech function using gTTS with retry logic
def text_to_speech(text, audio_file_path):
    retries = 5
    delay = 1
    for attempt in range(retries):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(audio_file_path)
            return audio_file_path
        except Exception as e:
            if '429' in str(e):
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            else:
                raise
    return audio_file_path

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        clients.remove(websocket)

async def send_progress(message: str):
    for client in clients:
        await client.send_text(message)

@app.post('/upload/')
async def upload_pdf(file: UploadFile = File(...), summarize: bool = Form(False), auto_scroll: bool = Form(False)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = f"uploads/{file.filename}"
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    await send_progress("10")  # 10%: File uploaded successfully

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(file_path)
    await send_progress("30")  # 30%: Extracted text from PDF

    # Split the text into manageable chunks
    chunks = split_text_into_chunks(pdf_text, chunk_size=2048)
    await send_progress("50")  # 50%: Split text into chunks

    if summarize:
        # Generate a summary for each chunk
        summaries = []
        for chunk in chunks:
            summary = generate_response("summarize: " + chunk)
            summaries.append(summary)
            await send_progress(f"{50 + (20 * len(summaries) / len(chunks)):.0f}")  # Update progress

        text_segments = summaries
    else:
        text_segments = chunks

    # Create directory for audio segments
    audio_dir = 'audio_segments'
    os.makedirs(audio_dir, exist_ok=True)

    # Save the text segments to a single audio file
    combined_audio = AudioSegment.empty()
    timestamps = []
    current_time = 0

    for index, segment in enumerate(text_segments):
        audio_file_path = os.path.join(audio_dir, f'segment_{index}.mp3')
        text_to_speech(segment, audio_file_path)
        segment_audio = AudioSegment.from_mp3(audio_file_path)
        combined_audio += segment_audio
        timestamps.append((current_time, current_time + len(segment_audio)))
        current_time += len(segment_audio)

        await send_progress(f"{70 + (30 * (index + 1) / len(text_segments)):.0f}")  # Update progress

    combined_audio_path = os.path.join(audio_dir, "combined_audio.mp3")
    combined_audio.export(combined_audio_path, format="mp3")

    await send_progress("100")  # 100%: Processing complete
    return {"summary": text_segments, "audio_file": "combined_audio.mp3",
            "timestamps": timestamps, "scroll": auto_scroll}

@app.get("/")
async def get(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

@app.get('/audio/{filename}')
async def get_audio_file(filename: str):
    file_path = os.path.join('audio_segments', filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/mpeg', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
