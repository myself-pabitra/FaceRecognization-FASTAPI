from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import face_recognition
from typing import List


app = FastAPI()

class Result(BaseModel):
    match: bool
    error: str = None


@app.post('/images', response_model=Result)
async def post_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        first_image = face_recognition.load_image_file(file1.file)
        second_image = face_recognition.load_image_file(file2.file)

        first_face_encodings = face_recognition.face_encodings(first_image)
        second_face_encodings = face_recognition.face_encodings(second_image)

        first_face_detected = len(first_face_encodings) > 0
        second_face_detected = len(second_face_encodings) > 0

        if not first_face_detected and not second_face_detected:
            return Result(match=False, error="Error: No face detected in both images")
        elif not first_face_detected:
            return Result(match=False, error="Error: No face detected in file1")
        elif not second_face_detected:
            return Result(match=False, error="Error: No face detected in file2")
        else:
            first_face_encoding = first_face_encodings[0]
            second_face_encoding = second_face_encodings[0]

            result = face_recognition.compare_faces([first_face_encoding], second_face_encoding)

            return Result(match=result[0])

    except FileNotFoundError:
        return Result(match=False, error="Error: File not found")

    except Exception as e:
        return Result(match=False, error=f"Error: {str(e)}")


