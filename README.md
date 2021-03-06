# LinguaSignans-ASL-Detection-System 
A Real-time Sign Language Detection System
Team Name: H2Y2
Team Members: Huimin Wang; Yining Wang; Yicheng Tao; Hanwen Guo

## Project Description
Our project targets recognizing American Sign Language (ASL) using computer vision techniques to help bridge the communication gap between sign language users and the others. We plan to design a recognition framework including image preprocessing, feature extraction, and classification. We will use ASL image dataset as the main subject, which includes about 35000 hand gesture images of ASL letters (​Sign Language MNIST (​https://www.kaggle.com/datamunge/sign-language-mnist​)​. Our final goal is to develop a real-time sign language detection system that provides two-way simultaneous translation between ASL and English. We will use OpenCV, PyTorch, along with other useful packages in Python to implement the algorithm. The system will be hosted by a web application based on JavaScript and HTML/CSS.

## Positive Impact on Society
Many language systems have developed their own sign language to assist communication among people with hearing impairment. Sign language attempts to remove the communication barrier between hearing-impaired people and others even when no hearing aids are present. Among all the sign languages, American Sign Language (ASL) is the most widespread language in the United States and the fourth-most studied second language at American universities. However, not every one of us understands ASL. People with hearing disabilities spend a considerable amount of time to learn ASL, but the language barrier still exists between individuals who use and understand ASL and those who don’t. The reason that we want to utilize Computer Vision to build a real-time sign language detection system is to remove this barrier completely, allowing everyone to communicate smoothly with one another. Technological advancements in Machine Learning and Computer Vision enable us to create a better simultaneous detection system that accurately recognizes ASL hand gestures. As accessibility is getting more and more attention in our society nowadays, our project aims to create a positive impact by supporting social inclusion especially for people with hearing disabilities.ystem

## Usage
Web application has only been tested on MacOS 10.15.7, Python 3.8.5, and Chrome 87.0.

In the target directory, run
```
$ pip install -r src/requirements.txt
```

To start the web application, run
```
$ cd src
$ python webstreaming.py --ip 0.0.0.0 --port 8000
```
The home page is running on http://0.0.0.0:8000/.

