from gradio_client import Client

client = Client("Flacco-33/EmotionAnalisisVideo")
result = client.predict(
		video_url="https://firebasestorage.googleapis.com/v0/b/edulytics-8a525.appspot.com/o/studentVideo%2Ftite.mp4?alt=media&token=63571264-c7cd-425c-8b52-a850778751e4",
		api_name="/predict"
)
print(result)