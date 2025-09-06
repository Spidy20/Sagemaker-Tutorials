## Tutorial - 4 Serverless Sklearn Inference with Lambda + API Gateway

![Tutorial Thumbnail](https://github.com/Spidy20/Sagemaker-Tutorials/blob/master/Tutorial%20-%203%20Sagemaker%20Build%20Custom%20Algorithm/yt_thumb.jpg)
[Watch the tutorial here ►](https://youtu.be/_OjFubgXcWQ)

### Overview

This tutorial guides you through the process of building a serverless inference for SKLearn ML model using API Gateway and AWS Lambda function using AWS SAM.

### Tutorial Content

In this tutorial, we cover the following steps:

**Step-1: Train the model (Local / SageMaker / On-Premises)**
Train your scikit-learn model on your chosen environment and capture baseline metrics for later validation.

**Step-2: Serialize the model (Pickle / Joblib)**
Export the fitted pipeline to a single artifact (e.g., `model-inference-1.0.joblib` or `.pkl`) with clear versioning.

**Step-3: Record runtime versions**
Note the exact Python and scikit-learn versions (and OS/arch if relevant) used during training to ensure Lambda runtime compatibility.

**Step-4: Implement the Lambda inference handler**
Place the model file in the **same directory** as the handler; load it via `pathlib`. Parse the API Gateway JSON body (`instances`), run inference, and return **Predictions**, **Probabilities** (single max per item), **Timestamp**, and **TatMs** with PascalCase keys. Add validation, error handling, and logging.

**Step-5: Prepare dependencies & SAM layout**
List only required libs (e.g., numpy, scikit-learn, joblib) in `requirements.txt`. Organize a SAM-compatible structure with `template.yaml`, an `API/` folder (handler + model + requirements), and an optional `events/` folder for sample payloads.

**Step-6: Define infrastructure in SAM**
Describe the Lambda (runtime, memory, timeout), attach a REST API route **POST `/inference`** with CORS, and optionally set environment variables (e.g., log level, class names). Ensure least-privilege IAM permissions.

**Step-7: Build and deploy the stack**
Perform a containerized build to produce Lambda-compatible artifacts, then deploy the SAM stack to your target AWS account.s.

**Step-8: Validate, monitor, and Test**
Call the `/inference` endpoint with representative inputs, verify response and test it.

### AWS Services Used
-----------------
- **AWS Lambda**              : Python 3.11 serverless runtime
- **Amazon API Gateway (REST)**  : /inference endpoint, CORS enabled
- **AWS SAM**                    : Build + deploy (with Docker for reproducible builds)
- **Amazon CloudWatch Logs**     : Function logs and troubleshooting
- **Cloudformation** : Manage SAM stack

### Commands & SAM Structure
# Project layout (SAM)

```text
ML-Inferenc-App/
├─ template.yaml
├─ API/
│  ├─ inference_lambda_handler.py      # Lambda handler (loads model via pathlib from same folder)
│  ├─ model-inference-1.0.joblib       # serialized sklearn model (joblib/pickle)
│  └─ requirements.txt                 # numpy, scikit-learn
```

#### Build Command
```shell
sam build --template-file template.yaml --use-container
```
#### Deploy Command
```shell
sam deploy --stack-name ml-inference \
  --resolve-s3 \
  --capabilities CAPABILITY_IAM \
  --region ap-south-1
```

### Support

If you find this tutorial helpful, please consider giving it a star⭐ or forking the repository to support our work.

- [Buy me a Coffee☕](https://www.buymeacoffee.com/spidy20)
- [Donate via PayPal (Your support inspires us to create more projects)](https://www.paypal.me/spidy1820)
- Donate me via UPI🇮🇳  - kushalbhavsar1@ybl