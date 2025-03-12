I saw a job posting on upwork that I would be a perfect fit for. I have already done something similar and I have a blog post on it as well as the code. I want you to make a proposal for the job in a nice markdown format.
Here are some of my requirements for the proposal:
- $30 per hour
- 1.5 months
- For those 1.5 months, write down the steps of what we'll do. From data gathering/labelling to model training, testing, deployment. So that it shows that (1) we know what we're doing and (2) 1.5 months is justified.
- Use the blog's content and code as well. I mean, don't include the actual lines of the code in the proposal, I mean as a reference for my skills and capability.

<job_posting>
Expert Needed for Football Computer Vision Model Training
I am seeking an experienced machine learning specialist to assist in training and fine-tuning an amateur football computer vision model. The ideal candidate will have a strong background and experience in computer vision as it relates to accurately detecting and tracking objects across frames. Your expertise will help improve model accuracy and performance to assist in building a robust system for detecting and tracking players, goalkeepers, ball and referees across video frames from amateur football game footage. If you have a passion for sports and cutting-edge technology, I would love to hear from you!
</job_posting>

<my_blog>
https://medium.com/@dreamai/ronaldo-messi-individual-highlights-maker-using-yolov8-detection-and-tracking-in-streamlit-0983a1ebeca6

# Ronaldo/Messi Individual Highlights Maker using YOLOv8 Detection and Tracking in Streamlit

Lionel Messi and Cristiano Ronaldo, synonymous with football excellence, have collectively shaped an era where their individual brilliance on the pitch transcends mere competition. Their extraordinary skills, stunning goals, and unmatched athleticism have not only etched their names in the archives of football history but have also given rise to a captivating spectacle that transcends the sport itself. The individual highlights of Messi and Ronaldo, capturing their mesmerizing goals and unparalleled feats, stand as some of the most watched and celebrated moments in the world of sports. If it wasn’t already apparent, I am an avid football enthusiast with the added perk of working in the field of artificial intelligence. Consequently, I made the deliberate choice to merge these two passions, resulting in an immensely enjoyable and fascinating endeavor. This synergy not only offers immense personal satisfaction but also presents a myriad of opportunities and potential monetary value. Without any more delay, let’s jump right into it.

# Problem Statement

Youtubers dedicated to sharing individual highlights of football maestros like Messi and Ronaldo often face a formidable challenge — the arduous task of manually sifting through a full 90-minute match to extract the key moments. This labor-intensive process not only demands considerable time and effort but can also be prone to oversight. The problem at hand is the need for an efficient solution to sift through the entirety of a match and distill it down to the most significant moments, as much as possible. Here, the integration of A.I. and machine learning emerges as a transformative remedy. These technologies can intelligently filter and identify the standout instances, allowing Youtubers to considerably expedite their editing workflow. By automating the extraction of highlights, we aim to alleviate the time burden on content creators. To address this, we’ll break it down into three phases. It’s time to kick-off!

# Phase 1: Dataset Preparation and Training the YOLOv8 Detection Model

# Dataset

***NOTE:*** Datasets were generated for both Messi and Ronaldo, but the training results for the latter proved significantly superior. Consequently, for the purpose of this blog, I will be utilizing Ronaldo’s dataset for our model. I intend to enhance Messi’s dataset and corresponding model in the near future. Both datasets are accessible, and you have the flexibility to choose either one.

Our objective is to identify Ronaldo within an in-game video, where other players are also present. To achieve this, we will employ three classes: “ronaldo,” “teammate,” and “opponent,” with the primary focus on the “ronaldo” label. For the creation and labeling of our dataset, we will leverage the [**RoboFlow**](https://roboflow.com/annotate) website’s platform. This platform facilitates efficient scaling of labeling tasks through a skilled workforce, advanced tools, and specialized expertise to swiftly and accurately label extensive datasets. The API from RoboFlow will generate a dataset comprising image frames extracted from videos, ensuring that all three labels or, at the very least, the “ronaldo” label is present.

To construct our dataset, we utilized individual highlight videos of Ronaldo, specifically three matches for Al-Nassr FC and three matches for the Portugal National Team, sourced from YouTube. Once the dataset is generated by RoboFlow, the next step is to label the data. Given that our model is based on the YOLOv8 detection model, we will employ the **Bounding Box Tool** to annotate our images, where each box corresponds to a specific class. In particular, each frame will feature a box representing “ronaldo,” as this is the pivotal class that our model needs to recognize. The utilization of individual highlight videos ensures the inclusion of this crucial class in our dataset. Additionally, RoboFlow offers the capability to assign batches to different team members, streamlining the annotation process for increased efficiency. Download any of the datasets from below:

[**Ronaldo Dataset**](https://universe.roboflow.com/dreamai/ronaldo-detection/dataset/3)

[**Messi Dataset**](https://universe.roboflow.com/dreamai/messi-detection/dataset/5)

# Training

Now that our dataset is ready, we can move on to training our model using a pre-trained **YOLOv8 detection model**. The training process is relatively straightforward, as outlined [here](https://docs.ultralytics.com/modes/train/). By using just a few lines of code, though it requires a significant amount of training time, we can effectively train our model. The following code snippet illustrates the process:

```
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data='/content/Ronaldo-Detection-3/data.yaml', epochs=300, patience=0)
```

This code encompasses the essential steps for training the model, including parameter specification, pre-trained model loading, and the initiation of the training loop with the prepared dataset. It’s important to note that the actual implementation may vary based on the library or framework chosen for this task.

# Phase 2: The Workflow with Helper Functions

Now that our model is trained, we enter the phase where the real magic happens. As mentioned earlier, since our model is designed for image detection, it cannot directly predict on a video. We need to extract frames from the video and then feed them to the model. Fortunately, YOLOv8 provides a *track* method, which takes a video as input and automatically predicts on a sequence of frames, essentially performing tracking. In theory, this aligns perfectly with our goal of tracking Ronaldo in a football match video. However, there are complexities involved.

One challenge is that the model will generate predictions for every frame, including those where Ronaldo is not present. The resulting video will have bounding boxes around various classes, even when irrelevant. Additionally, video processing consumes a considerable amount of memory, making the straightforward use of **yolo.track** insufficient for our desired outcome.

But worry not, as this is where the magical part of the process comes into play. We’ll need to write and modify code around the track method to achieve our specific goal. While it may seem complicated and overwhelming, we can streamline the process by defining a workflow with steps, which will be implemented through the creation of functions. These functions will orchestrate the necessary steps to filter and process the predictions, ensuring that we obtain the desired outcome of accurately tracking Ronaldo in the football match video.

*Workflow Of The Application*

# Step 1: Divide large input video into parts

As mentioned earlier, dealing with large videos can lead to memory issues, necessitating the division of the video into smaller sub-videos to ensure a smooth execution of the entire process. To achieve this, we will utilize the [**MoviePy**](https://zulko.github.io/moviepy/) library in Python, which offers various methods to facilitate video processing. One crucial step in this process is determining the duration of the original video, and MoviePy provides a solution for this. Below is a function that completes our initial step:

```
def split_vid(video, split_path):
    video = VideoFileClip(str(video))
    duration = video.duration
    split = duration / 10.0
    c = 0
    i = 0
    while i < duration:
        if (i+split) > duration:
            sub_clip = video.subclip(i,duration)
            try:
                sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            except IndexError:
                try:
                    clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264",
                                        audio=False)
                except Exception as e:
                    logger.info("exception caught: %s" % e)
            i = i+split
            c = c+1
        else:
            sub_clip = video.subclip(i,i+split)
            sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            i = i+split
            c = c+1
```

# Step 2: Run Model Tracking on Sub-Videos

Now, we will leverage the capabilities of **yolo.track** and apply it to the sub-videos generated in the previous step. The model provides prediction objects for each image frame in the video. Our objective is to extract frames where a specific class, such as *ronaldo*, has been predicted, disregarding the others. Essentially, we aim to narrow down the original video to scenes featuring Ronaldo. Upon scrutinizing the objects returned by our trained model, we’ve discovered that they include the original image frame dimensions without the bounding boxes. This is precisely what we need. Using these images, we can construct a video utilizing the [**OpenCV**](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) library (*a helpful tip for optimizing memory usage instead of relying on MoviePy once again*). Initially, let’s establish a function to filter the model results to the essential image frames, a pivotal step in the process.

```
def filter_vid(results,fps):
    '''
    results : The list of objects that the model returns
    fps : The fps (frames per second) of the original video

    This function will return a list of images after filtering the original results objects
    '''
    imgs=[]
    i=0
    fps = int(fps)
    step = int(fps*2)
    while i < len(results):
        #img = results[i].orig_img
        cls = list(results[i].boxes.cls)
        if cls != []:
            temp = []
            c=0
            for x in range(i,i+fps):
                if i+fps > len(results):
                    break
                temp.append(results[x].orig_img)
                t_cls = list(results[x].boxes.cls)
                if t_cls == []:
                    c=c+1
            if c <= 15:
                for f in range(i+fps, i+step):
                    if i+fps > len(results) or i+step > len(results):
                        break
                    temp.append(results[f].orig_img)
                imgs+=temp
                i = i+step
            else:
                i = i+fps
        else:
            i = i+1

    return imgs
```

Next, let’s create another function that will make the video using the image frames that the **filter\_vid** function provided.

```
def create_vid(img_list, dest_path, x, fps):
    '''
    img_list : The list of image frames
    dest_path : The path where the resulting video will be stored
    x : a count to name the video in asencding order
    fps : The fps (frames per second) of the original video
    '''
    size = (720, 1280)
    vid_name = dest_path+'wowTest'+str(x)+'.mp4'
    temp = 'wowTest'+str(x)+'.mp4'
    out = cv2.VideoWriter(vid_name,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,(size[1],size[0]),
                          isColor=True)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()
```

Using these two auxiliary functions, we will now harness the capabilities of the model and proceed to accomplish the second step of the workflow. While I will be loading my model weights, you have the option to load your own trained model if preferred.

```
#these are custom model weights, you can load your own model weights
model = YOLO('./ron_3k.pt')
split_vids = get_files(SUB_VIDS_PATH)
split_vids = sort_list(split_vids)
#split_vids is the list of the subvideos we created in the first step.
x=0
for sv in split_vids:
    #you can play around with the conf (confidence) value
    results = model.track(source=sv, persist=True, classes=1,
                          conf=0.65, tracker="bytetrack.yaml", save=False, show=False,
                          verbose=False, save_txt=False)
    results = filter_vid(results,fps)
    create_vid(results, HIGH_VIDS_PATH, x, fps)
    x = x+1
```

# Step 3: Concatenating the filtered videos to create the Final Video

After obtaining the filtered videos, the last step involves merging these videos into a singular video, constituting our final output and concluding the entire workflow. Once more, **OpenCV** will be employed for this concatenation process.

```
def concatenate_videos(dest_path, new_video_path):

    '''
    dest_path: The path where the filtered videos are stored
    new_video_path: The path where the final video will be stored
    '''

    highVids = get_files(dest_path)
    #li = sort_list(li)
    for i in range(len(highVids)):
        highVids[i] = './'+str(highVids[i])
    highVids.sort()
    size = (720, 1280)
    video = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*"MPEG"), 30, (size[1],size[0]))

    for v in range(len(highVids)):
        curr_v = cv2.VideoCapture(highVids[v])
        while curr_v.isOpened():
            r, frame = curr_v.read()
            if not r:
                break
            video.write(frame)

    video.release()
```

# Phase 3: Putting it all together in Streamlit

By combining Phases 1 and 2, we’ve successfully transformed a regular match video into a condensed version, predominantly featuring highlights of Ronaldo. Yet, what if we desire to showcase our model and the entire workflow accomplished in Phase 2? This is where [**Streamlit**](https://docs.streamlit.io/) becomes invaluable. Streamlit is an open-source Python library designed for effortlessly creating and sharing customized web apps tailored for machine learning and data science. With just a few lines of code, we can craft a straightforward Streamlit application, enabling the swift development and deployment of robust data apps.

# Putting all functions together into a single py file

In Phase 2, a series of functions were defined, including several helper functions, to facilitate the achievement of our objective. To enhance code clarity and promote reusability, we plan to consolidate all these functions into a single file named **functions.py**. This file will be subsequently imported into our Streamlit application file, streamlining the overall structure and improving code organization.

# Creating the Streamlit application

Alright, time to create our Streamlit application. It can be imported just like any other Python libary.

```
import streamlit as st
```

Upon importing, a multitude of functions and widgets become accessible. For instance, to set the title of the page, we can utilize st.title(). Similarly, incorporating text is straightforward with st.write(). The Streamlit [documentation](https://docs.streamlit.io/) offers comprehensive insights into the functionality of various tools available.

To construct the foundation of our application, we’ll primarily rely on three of Streamlit’s tools: **st.file\_uploader** **st.button** **st.download\_button**

Let’s briefly outline the functionality of each:

## st.file\_uploader

In the previously defined workflow in Phase 2, the assumption was that the input video, a 45-minute half of a football game, was already available. However, in the context of an application, we aim to allow users to upload the video themselves. Streamlit provides a valuable tool for this purpose, namely **st.file\_uploader**. This tool enables users to upload any type of file, including video files (e.g., mp4, avi), and returns an UploadedFile object. By storing this uploaded file (video) in a designated folder, we can then display it to the user using st.video. It’s worth noting that uploaded files are subject to a default limit of 200MB. However, this limit can be adjusted by configuring the maxFileSize option when launching our application, providing flexibility based on specific requirements.

## st.button

The st.button function in Streamlit serves as a widget for displaying a button. It operates as an event handler, triggering the next set of actions when the user clicks on the button. In our application, we plan to present the user with a button labeled **“Click Here to Generate Ronaldo Highlights”** If the user has successfully uploaded the input video, clicking this button will initiate the execution of our defined workflow. Conversely, if no video has been uploaded, the button will prompt the user to first upload a video before proceeding with the workflow. This interactive button enhances user engagement and controls the flow of the application based on user inputs.

## st.download\_button

Upon the completion of the entire program execution, the application will prompt the user to download the resulting video. This download functionality can be implemented using the **st.download\_button** provided by Streamlit. By incorporating this button, users can conveniently download the processed video, representing the final output of the application. This feature enhances user experience by providing a straightforward means to obtain and save the generated content.

# The Streamlit App Code

Putting it all together

```
import streamlit as st
from functions import *

INPUT_VIDEO_PATH = "./input_vid"
SUB_VIDS_PATH = "./sub_vids"
HIGH_VIDS_PATH = "./high_vids"

#this will create 3 folders on the paths defined above
os.makedirs(INPUT_VIDEO_PATH, exist_ok=True)
os.makedirs(SUB_VIDS_PATH, exist_ok=True)
os.makedirs(HIGH_VIDS, exist_ok=True)

st.title("Ronaldo Detection")
st.write("Upload a video")
uploaded_file = st.file_uploader("File Uploader for Ronaldo highlights",type=['mp4', 'avi'])

if uploaded_file != None:
    with open(os.path.join(INPUT_VIDEO_PATH,uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())

    src_vid = get_files(INPUT_VIDEO_PATH)
    st.write(src_vid)
    vid = src_vid[0]
    st.write(vid)
    file_stats = os.stat(vid)
    st.write(file_stats.st_size / (1024 * 1024))

    video_file = open(vid, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

if st.button("Click here to make Ronaldo Highlights"):

        if uploaded_file is not None:
            #these are custom model weights, you can load your own model weights
            model = YOLO('./ron_3k.pt')
            #Just to make sure at which index the 'ronaldo' class is placed, in this case it's 1
            st.write(model.names)
            fps = get_fps(vid)
            split_vid(vid, SUB_VIDS_PATH)
            split_vids = get_files(SUB_VIDS_PATH)
            split_vids = sort_list(split_vids)
            x=0
            for sv in split_vids:
                #you can play around with the conf (confidence) value
                results = model.track(source=sv, persist=True, classes=1,
                          conf=0.65, tracker="bytetrack.yaml", save=False, show=False,
                          verbose=False, save_txt=False)
                results = filter_vid(results,fps)
                create_vid(results, HIGH_VIDS_PATH, x, fps)
                x = x+1
            concatenate_videos(HIGH_VIDS_PATH, fps, "./finalVid.mp4")

            t_video_file = open("./finalVid.mp4", 'rb')
            t_video_bytes = t_video_file.read()
            st.video(t_video_bytes)
            with open("./finalVid.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="Final.mp4",
                        mime="video/mp4"
                      )

        else:
            st.write("Please, upload a video first")
```

To launch the streamlit app, run this command:

```
streamlit run ronaldo.py --server.maxUploadSize 2000
```
</my_blog>

<my_blog_code>
================================================
File: functions.py
================================================
import os
import gc
import torch
from pathlib import Path
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import glob
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sort_list(paths_list):
    paths_list = [str(v) for v in paths_list]
    paths_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return paths_list
    

def setify(o):
    return o if isinstance(o, set) else set(list(o))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path, extensions=None, recurse=True, folders=None, followlinks=True, make_str=False
):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    if folders is None:
        folders = list([])
    path = Path(path)
    if extensions is not None:
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    if make_str:
        res = [str(o) for o in res]
    return list(res)

def del_fol (file_path) :
    folder = file_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            

def filter_vid(results,fps):
    imgs=[]
    i=0
    fps = int(fps)
    step = int(fps*2)
    while i < len(results):
        #img = results[i].orig_img
        cls = list(results[i].boxes.cls)
        if cls != []:
            temp = []
            c=0
            for x in range(i,i+fps):
                if i+fps > len(results):
                    break
                temp.append(results[x].orig_img)
                t_cls = list(results[x].boxes.cls)
                if t_cls == []:
                    c=c+1
            if c <= 15:
                for f in range(i+fps, i+step):
                    if i+fps > len(results) or i+step > len(results):
                        break
                    temp.append(results[f].orig_img)
                imgs+=temp
                i = i+step
            else:
                i = i+fps
        else:
            i = i+1
            
    return imgs
            
def create_vid(img_list, dest_path, x, fps):
    size = (720, 1280)
    vid_name = dest_path+'wowTest'+str(x)+'.mp4'
    temp = 'wowTest'+str(x)+'.mp4'
    out = cv2.VideoWriter(vid_name,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,(size[1],size[0]),
                          isColor=True)

    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()
    #os.system("ffmpeg -i highVids/wowTest0.mp4 -vcodec libx264 compVids/wowTest0.mp4")
    
    
def split_vid(video, split_path):
    video = VideoFileClip(str(video))
    half = (video.duration / 2.0) - 6
    split = half / 10.0
    c = 0
    i = 0
    #clips = []
    while i < half:
        if (i+split) > half:
            #diff = abs((i+split) - half)
            sub_clip = video.subclip(i,half)
            try:
                sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            except IndexError:
                try:
                    clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264",
                                        audio=False)
                except Exception as e:
                    logger.info("exception caught: %s" % e)
            #sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            i = i+split
            c = c+1
        else:
            sub_clip = video.subclip(i,i+split)
            sub_clip.write_videofile(split_path+"sub_clip"+str(c)+".mp4", codec="libx264")
            #clips.append(sub_clip)
            i = i+split
            c = c+1
        
def get_fps(video):
    video = VideoFileClip(str(video))
    fps = video.fps
    return fps
        
def final_vid(dest_path):
    highVids = get_files(dest_path)
    highVids = sort_list(highVids)
    for i in range(len(highVids)):
        highVids[i] = VideoFileClip(highVids[i])
    video = concatenate_videoclips(highVids)
    video.write_videofile("./final.mp4", codec="libx264")
    
def concatenate_videos(dest_path, fps, new_video_path):
    highVids = get_files(dest_path)
    #li = sort_list(li)
    for i in range(len(highVids)):
        highVids[i] = './'+str(highVids[i])
    highVids.sort()
    size = (720, 1280)
    video = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*"MPEG"), fps, (size[1],size[0]))

    for v in range(len(highVids)):
        curr_v = cv2.VideoCapture(highVids[v])
        while curr_v.isOpened():
            r, frame = curr_v.read()
            if not r:
                break
            video.write(frame)

    video.release()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



================================================
File: ronaldo.py
================================================
import streamlit as st
from functions import *

INPUT_VIDEO_PATH = "./input_vid" 
SUB_VIDS_PATH = "./sub_vids"
HIGH_VIDS_PATH = "./high_vids"

#this will create 3 folders on the paths defined above
os.makedirs(INPUT_VIDEO_PATH, exist_ok=True)
os.makedirs(SUB_VIDS_PATH, exist_ok=True)
os.makedirs(HIGH_VIDS, exist_ok=True)

st.title("Ronaldo Detection")
st.write("Upload a video")
uploaded_file = st.file_uploader("File Uploader for Ronaldo highlights",type=['mp4', 'avi'])


if uploaded_file != None:
    with open(os.path.join(INPUT_VIDEO_PATH,uploaded_file.name),"wb") as f:    
        f.write(uploaded_file.getbuffer())

    src_vid = get_files(INPUT_VIDEO_PATH)
    st.write(src_vid)
    vid = src_vid[0]
    st.write(vid)
    file_stats = os.stat(vid)
    st.write(file_stats.st_size / (1024 * 1024))

    video_file = open(vid, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    

    
if st.button("Click here to make Ronaldo Highlights"):

        if uploaded_file is not None:
            model = YOLO('./ron_3k.pt') #these are custom model weights, you can load your own model weights
            st.write(model.names) #Just to make sure at which index the 'ronaldo' class is placed, in this case it's 1
            fps = get_fps(vid)
            split_vid(vid, SUB_VIDS_PATH)
            split_vids = get_files(SUB_VIDS_PATH)
            split_vids = sort_list(split_vids)
            x=0
            for sv in split_vids:
                #you can play around with the conf (confidence) value
                results = model.track(source=sv, persist=True, classes=1,
                          conf=0.65, tracker="bytetrack.yaml", save=False, show=False,
                          verbose=False, save_txt=False) 
                results = filter_vid(results,fps)
                create_vid(results, HIGH_VIDS_PATH, x, fps)
                x = x+1
            concatenate_videos(HIGH_VIDS_PATH, fps, "./finalVid.mp4")
            
            t_video_file = open("./finalVid.mp4", 'rb')
            t_video_bytes = t_video_file.read()
            st.video(t_video_bytes)
            with open("./finalVid.mp4", "rb") as file:
                btn = st.download_button(
                        label="Download video",
                        data=file,
                        file_name="Final.mp4",
                        mime="video/mp4"
                      )
  
        else:
            st.write("Please, upload a video first")
</my_blog_code>