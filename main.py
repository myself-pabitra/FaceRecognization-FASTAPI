from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import face_recognition


app = FastAPI()

class Result(BaseModel):
    match: bool
    error: str = None

@app.post('/images', response_model=Result)
async def post_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        first_image = face_recognition.load_image_file(file1.file)
        second_image = face_recognition.load_image_file(file2.file)

        first_face_encoding = face_recognition.face_encodings(first_image)[0]
        second_face_encoding = face_recognition.face_encodings(second_image)[0]

        result = face_recognition.compare_faces([first_face_encoding], second_face_encoding)

        return Result(match=result[0])

    except FileNotFoundError:
        return Result(match=False, error="Error: File not found")

    except IndexError:
        return Result(match=False, error="Error: No face found in one or both images")

