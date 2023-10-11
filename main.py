from fastapi import FastAPI, Form
import face_recognition

app = FastAPI()

@app.post('/images')
async def post_images(name1: str = Form(...), name2: str = Form(...)):
    # map the url to the ones in the folder images
    first_image_path = "images/" + name1
    second_image_path = "images/" + name2

    # loading the image inside a variable
    first_image = face_recognition.load_image_file(first_image_path)
    second_image = face_recognition.load_image_file(second_image_path)

    # encode the images
    first_face_encoding = face_recognition.face_encodings(first_image)[0]
    second_face_encoding = face_recognition.face_encodings(second_image)[0]

    # compare the two encoded images
    result = face_recognition.compare_faces([first_face_encoding], second_face_encoding)

    # returning the final result
    if result[0] == True:
        return "True"
    else:
        return "False"
