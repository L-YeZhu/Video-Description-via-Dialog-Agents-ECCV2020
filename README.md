## Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents - ECCV 2020
This repository is the implementation for the unseen video description task introduced in the paper Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents. Our codes are based on [AudioVisualSceneAwareDialog(Hori et. al.)](https://github.com/dialogtekgeek/AudioVisualSceneAwareDialog) and [Baseline on AVSD(Schwartz et. al.)](https://github.com/idansc/simple-avsd), we thank the authors of the previous work to share their data and codes.


#### 1. Introduction of the task
We introduce a task that aims whose ultimate goal is for one coversational agent to describe an unseen video based on the dialog and two static frames from the video as shown below.
<p align="center">
<img src="https://github.com/L-YeZhu/AVSD-Agents/blob/master/figures/fig1.png" width="500">
  </p>

#### 2. Required packages
- python 2.7
- pytorch 0.4.1
- Numpy
- six
- java 1.8.0

#### 3. Data
The original AVSD dataset used in our experiments can be found [here](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge).\
The annotations can be downloaded [here](https://drive.google.com/file/d/1CWXNeXsxz8UelosF8XWU9bNLFkUON2J3/view?usp=sharing).
The audio-visual features can be downloaded [here](https://drive.google.com/file/d/15KwizowgEtUJKESDOGEICutHrqiXFQ5e/view?usp=sharing).
