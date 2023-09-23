## Tutorial - 3 Building a Custom Algorithm in Amazon SageMaker

![Tutorial Thumbnail](https://github.com/Spidy20/Sagemaker-Tutorials/blob/master/Tutorial%20-%203%20Sagemaker%20Build%20Custom%20Algorithm/yt_thumb.jpg)
[Watch the tutorial here ►](https://youtu.be/_OjFubgXcWQ)

### Overview

This tutorial guides you through the process of building a custom algorithm in Amazon SageMaker using the SM-Docker template. You will learn how to create and deploy your own machine learning algorithm.

### Tutorial Content

In this tutorial, we cover the following steps:

1. **Download Official SageMaker Structure**: Start by obtaining the official SageMaker project structure.

2. **Edit `train` & `predictor.py`**: Customize the `train` and `predictor.py` files to match your specific algorithm requirements.

3. **Add Required Package Dependencies in Dockerfile**: Modify the Dockerfile to include any necessary package dependencies for your custom algorithm.

4. **Organize Files in SageMaker Studio File-System (EFS)**: Arrange all files and directories with the proper structure within the SageMaker Studio file-system, specifically in Amazon Elastic File System (EFS).

5. **Execute Notebook and Create Algorithm Container with `sm-docker`**: Use a Jupyter notebook and the `sm-docker` command to create an algorithm container.

6. **Train Model with Custom Algorithm**: Utilize the custom algorithm container(we have created in step-5) to train a machine learning model.

7. **Create SageMaker Endpoint and Test the Model**: Set up a SageMaker Endpoint and evaluate the model's performance using test data.

### AWS Services Used

The tutorial utilizes the following AWS services:

- **SageMaker Studio (Jupyter Notebook)**: We use SageMaker Studio for notebook execution and development.

- **ECR (Elastic Container Registry)**: To store the custom algorithm image.

- **CodeBuild**: Used for SM-Docker building activities and execution.

### Support

If you find this tutorial helpful, please consider giving it a star⭐ or forking the repository to support our work.

- [Buy me a Coffee☕](https://www.buymeacoffee.com/spidy20)
- [Donate via PayPal (Your support inspires us to create more projects)](https://www.paypal.me/spidy1820)

