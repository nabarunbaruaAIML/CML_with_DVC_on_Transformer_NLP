# Continuous Machine Learning on Huggingface Transformer with DVC including Weights & Biases implementation.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FnabarunbaruaAIML%2FCML_with_DVC_on_Transformer_NLP&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Main idea of this project is to explain how CML can be implemented in NLP Project therefore main focus of this Project is to explain how CML can be implemented. We assume that user is well verse in Transformer, DVC & W&B Implementation.

Below are the online resource used for building this Project.

- [DVC Youtube Playlist](https://www.youtube.com/playlist?list=PL7WG7YrwYcnDBDuCkFbcyjnZQrdskFsBz) and [DVC CML Official Doc](https://cml.dev/doc/self-hosted-runners?tab=AWS)
- [Official Huggingface Website](https://huggingface.co/docs)
- [Weights & Biases Official Doc](https://docs.wandb.ai/guides/integrations/huggingface)
- Apart from above from iNeuron's Team Sunny Bhaveen Chandra Session on DVC Helped to complete this project

Before we begin with the session few things which need to be setuped are as follows:
- One AWS IAM user with EC2 & S3 Developer Access 
- S3 Bucket to store the Dataset
- Second EC2 Spot Instance need to be requested in advance if not done earlier.

Please follow these online Resource AWS related information
![image]({https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white})
- [Youtube Resource 1](https://www.youtube.com/watch?v=rYHt0gtRKFg&t=180s)
- [Youtube Resource 2](https://www.youtube.com/watch?v=GCt-cymgdvo)
- [For right GPU Selection](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86) and for the same [Youtube Link](https://www.youtube.com/watch?v=4bVrIbgGWEA)

In this project we want to implement Transformer Classification for [Kaggle Dataset](https://www.kaggle.com/hassanamin/atis-airlinetravelinformationsystem), idea is to implement DVC so that from DVC studio we can do the experiments, where as in Transformers Weights & Biases have in built implementation which allows to save Best Model's weights & Metrices. 

We can use DVC for Metric tracking but for that further changes & implementation need to be implemented. On the other hand, Weights & Biases just need minimalist changes in any transformer code to start tracking. One major advantage which I see in using Weights & Biases i.e. it save best model which otherwise we had to do after every experiments.

Now I believe that we're through with the goal and clear vision as in what we want to do in this project. 

Lets begin by click and going into this [Template Repository](https://github.com/nabarunbaruaAIML/project-template-with-DVC) 

Once into Template Repository Please click Button **Use this Template**.

![Template Repository](./Template-Repo.jpg)

then fill Details and create repository from the template.
![Fill Details](./Fill-Details.jpg)



## Now follow below steps.


### STEP 01- Create local repository after cloning the repository. We can use git bash for cloning if using windows system. If Linux/Mac OS then Terminal will work.

### STEP 02- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```
Activate the environment in the VSCode by executing the following command:
```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 03- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 04- initialize the dvc project
```bash
dvc init
```

### STEP 05- commit and push the changes to the remote repository

# Work in Progress
