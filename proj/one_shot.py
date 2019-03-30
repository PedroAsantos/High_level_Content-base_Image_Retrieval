import face_recognition
import cv2
#encode
img1 = cv2.imread("faces/harrison.jpeg")[:, :, ::-1]
img1_enc = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("faces/pitt.jpeg")
img2_enc = face_recognition.face_encodings(img2)[0]

img3 = face_recognition.load_image_file("faces/morgan.jpeg")
img3_enc = face_recognition.face_encodings(img3)[0]
#recon
#unk_image = face_recognition.load_image_file("faces/harrison2.jpeg")
unk_image = cv2.imread("faces/harrison3.jpeg")[:, :, ::-1]

face_locations = face_recognition.face_locations(unk_image)
face_encodings = face_recognition.face_encodings(unk_image, face_locations)

for face_encoding in face_encodings:
    match = face_recognition.compare_faces([img1_enc, img2_enc, img3_enc], face_encoding, tolerance=0.60)
    print(match)
    print(face_recognition.face_distance([img1_enc, img2_enc, img3_enc], face_encoding))
