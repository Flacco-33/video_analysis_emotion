import os
import urllib.parse

import torch
import gradio as gr
import requests
import tempfile
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from emotion import detect_emotion, init

# Lista de todas las emociones
ALL_EMOTIONS = ['happy', 'sad', 'anger', 'surprise', 'neutral', 'disgust', 'fear', 'contempt']

def extract_video_name(url):
    parsed_url = urllib.parse.urlparse(url)
    video_name = os.path.basename(parsed_url.path)
    return video_name

def download_video(url):
    response = requests.get(url, stream=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_file.name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return temp_file.name

def detect(opt):
    source, imgsz = opt.source, opt.img_size
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'
    
    # Inicializar el diccionario con todas las emociones posibles en 0
    emotion_dict = {emotion: 0 for emotion in ALL_EMOTIONS}
    
    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                images = []
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    images.append(im0[int(y1):int(y2), int(x1):int(x2)])
                if images:
                    emotions = detect_emotion(images)
                    emotion_dict = count_emotions(emotions, emotion_dict)
    return emotion_dict

def count_emotions(emotions, emotion_dict):
    for emotion in emotions:
        label = emotion[0].split()[0]
        if label in emotion_dict:
            emotion_dict[label] += 1
        else:
            emotion_dict[label] = 1
    return emotion_dict


def gradio_detect(video_url):
    video_path = download_video(video_url)
    video_name = extract_video_name(video_url)
    try:
        class Opt:
            source = video_path
            img_size = 512
            conf_thres = 0.5
            iou_thres = 0.45
            device = ''
            augment = False
            agnostic_nms = False
        opt = Opt()
        emotion_dict = detect(opt)
    finally:
        os.remove(video_path)
    return {"video_name": video_name, "emotions": emotion_dict}

demo = gr.Interface(
    fn=gradio_detect,
    inputs=gr.Textbox(label="Video URL"),
    outputs=gr.Json()
)

if __name__ == '__main__':
    demo.launch()