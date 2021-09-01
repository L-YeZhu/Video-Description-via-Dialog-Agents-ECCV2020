## Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents - ECCV 2020
This repository is the implementation for the video description task introduced in the paper [Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents](https://arxiv.org/pdf/2008.07935.pdf). Our codes are based on [AudioVisualSceneAwareDialog(Hori et. al.)](https://github.com/dialogtekgeek/AudioVisualSceneAwareDialog) and [Baseline on AVSD(Schwartz et. al.)](https://github.com/idansc/simple-avsd), we thank the authors of the previous work to share their data and codes.

### Update 08/2021
We have published an extended version of this video description work at TPAMI with novel settings and experiments. You can check the paper [Saying the Unseen: Video Descriptions via Dialog Agents](https://arxiv.org/pdf/2106.14069.pdf). The code will be updated at this repo.



#### 1. Introduction of the task
We introduce a task whose ultimate goal is for one coversational agent to describe an unseen video based on the dialog and two static frames from the video as shown below.
<p align="center">
<img src="https://github.com/L-YeZhu/AVSD-Agents/blob/master/fig1.png" width="500">
  </p>

#### 2. Required packages
- python 2.7
- pytorch 0.4.1
- Numpy
- six
- java 1.8.0

#### 3. Data
The original AVSD dataset used in our experiments can be found [here](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge).\
The annotations can be downloaded [here](https://drive.google.com/file/d/1CWXNeXsxz8UelosF8XWU9bNLFkUON2J3/view?usp=sharing). Please extract to ‘data/’.\
The audio-visual features can be downloaded [here](https://drive.google.com/file/d/15KwizowgEtUJKESDOGEICutHrqiXFQ5e/view?usp=sharing). Please extract to ‘data/charades_features’.

#### 4. Running the code and pre-trained models
Use the command <code>./qa_run.sh</code> to run the codes.\
The codes are running under 4 different stages: evaluation tool prepration, training, inference and scores calculating. Note that to compute the SPICE scores, please follow the additional instructions from the [coco-pation project](https://github.com/tylin/coco-caption).\
The pretained model is available [here](https://drive.google.com/file/d/1wsOlG9HxJSotPpOVpQ_CwnLAD5KY6f9k/view?usp=sharing).

#### 5. Citation
Please consider citing our papers if you find them useful.
```
@InProceedings{zhu2020describing,    
  author = {Zhu, Ye and Wu, Yu and Yang, Yi and Yan, Yan},    
  title = {Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents},    
  booktitle = {The European Conference on Computer Vision (ECCV)},    
  year = {2020} 
  }
  
@InProceedings{zhu2021saying,    
  author = {Zhu, Ye and Wu, Yu and Yang, Yi and Yan, Yan},    
  title = {Saying the Unseen: Video Descriptions via Dialog Agents},    
  booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},    
  year = {2021}
  }
```
