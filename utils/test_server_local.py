import requests

# Server URL
url = "http://localhost:8080/predict"

# Image file to send
image_path = "/home/mzingerenko/Desktop/LecturesProject_Example/data/mnist_img3.png"
with open(image_path, "rb") as img_file:
    headers = {"Content-Type": "image/jpeg"}
    response = requests.post(url, headers=headers, data=img_file)

# Print the server response
print("Response Code:", response.status_code)
print("Response JSON:", response.json())
