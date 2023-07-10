# Generative AI - PaLM-2 model deployment in Cloud Run 

The post shows a frontend in [Gradio](https://gradio.app/) that exposes one of the PaLM-2 foundational models, `text-bison@001`, deployed in Cloud Run.

`text-bison@001` is [one of the foundational models](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models) based on PaLM-2 that is available in Vertex AI. This post shows a front-end exposing this model and [its main parameters](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#parameter_definitions) (temperature, output tokens, top-P and top-K) via a Gradio app. 

The model `text-bison@001` can be applied to use cases like dialog summarization, text generation, scoring for marketing, and many others.

The frontend is deployed through a Gradio app deployed in [Cloud Run](https://cloud.google.com/run). An screenshot of the app follows:

![LLM Text demo](images/text-demo.png)


## PaLM-2 in Vertex AI

PaLM-2 for Text is one of the foundational models available in Vertex AI. PaLM-2 technical erport can be found [here](https://ai.google/static/documents/palm2techreport.pdf). Using the Vertex AI SDK, you can easily call a prediction with the following:

```py
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextGenerationModel.from_pretrained("text-bison@001")
model.predict(
    prompt,
    max_output_tokens=max_output_tokens, # default 128
    temperature=temperature, # default 0
    top_p=top_p, # default 1
    top_k=top_k) # default 40
```

## Prompt examples

You can test the following propmt examples:

Prompt example for **dialog summarization**:
```
Provide a laconic summary for the following conversation:

Customer Service Rep: Hello, thank you for calling customer service. How can I help you today?    
Customer: I\'d like to return a product that I purchased.     
Customer Service Rep: Sure, I can help you with that. What is the item that you would like to return?      
Customer: I would like to return a [product name].      
Customer Service Rep: Okay, I can see that you purchased this on [date].     
Customer: Yes, that\'s correct.      
Customer Service Rep: Unfortunately, the return date for this product was on [date].     
Customer: Yes, I know. I\'ve been very busy and meant to return it sooner.     
Customer Service Rep: I\'m really sorry, but there\'s nothing that I can do about it.      
Customer: But I\'m not happy with the product. It\'s damaged.     
Customer Service Rep: I understand that you\'re not happy with the product. However, the return date for this product has passed.     
Customer: But I\'m not the only customer who\'s had this problem. There are other customers who have returned this product because it\'s damaged.     
Customer Service Rep: I\'m sure that there are other customers who have returned this product because it\'s damaged. However, the return date for this product has passed.      
Customer: I\'m going to write a review about this product and how your company doesn\'t stand behind its products.     
Customer Service Rep: I\'m sorry to hear that you\'re unhappy with the product. However, I cannot help you with this return. 
```

Prompt example to generate **finance info for an analyst report**:
```
You are an equities analyst researching information for your report with relevant facts and figures.

Tell me about the mortgage market in US.
```

Prompt example for **content generation for a marketing and social media campaign**:
```
We want to create a multi-media campaign that highlights the simple ingredients and sustainable sourcing practices from
our most popular granola bar. We want to highlight:
* We use only 6 simple ingredients: organic oats, organic almonds, local honey, organic whole flour, dried organic blackberries, free-range eggs
* Our packaging is made of 100% compostable materials
* We use rainwatrer harvesting in our oat and almond farming to redice the need of irrigation.
* We use solar energy to power irrigation systems and other farm equipments.
* Whenever we can, we support local farmers

Blog Headline:

Blog Post:

Instagram Caption:

Instagram Hastags:
```

Prompt example for **marketing campaign of clustered customers**:
```
Pretend you're a creative strategist, given the following clusters come up with creative brand persona and title labels for each of these clusters, and explain step by step; what would be the next marketing step for these clusters:

cluster 1, average spend $43.51, count of orders per person 1.23, days since last order 295.33
cluster 2, average spend $47.53, count of orders per person 1.23, days since last order 820.65
cluster 3, average spend $58.42, count of orders per person 3.51, days since last order 287.5
cluster 4, average spend $200.37, count of orders per person 1.17, days since last order 421.67
cluster 5, average spend $59.29, count of orders per person 3.49, days since last order 794.72
```


## User managed service account for Cloud Run

Since the application is deployed in Cloud Run, it uses the permissions of the compute service account by default. It's recommended to use a separate service account for minimum permissions. To do that, [create the service account with impersonation](https://cloud.google.com/run/docs/securing/service-identity) and the following two extra roles: `roles/aiplatform.user` to be able to call predictions and `roles/logging.logWriter` to be able to write logs.

```sh
PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
USER_ACCOUNT=<REPLACE_WITH_YOUR_USER_ACCOUNT>
# Create service account
gcloud iam service-accounts create cloud-run-llm \
    --description="Service account to call LLM models from Cloud Run" \
    --display-name="cloud-run-llm"

# add aiplatform.user role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloud-run-llm@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# add logging.logWriter role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:cloud-run-llm@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

# add permission to impersonate the sa (iam.serviceAccounts.actAs), since this is a user-namaged sa
gcloud iam service-accounts add-iam-policy-binding \
    cloud-run-llm@$PROJECT_ID.iam.gserviceaccount.com \
    --member="user:$USER_ACCOUNT" \
    --role="roles/iam.serviceAccountUser"
```


## Build and deploy in Cloud Run

To build and deploy the [Gradio app](https://gradio.app/) in [Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container), you need to build the Docker image in Artifact Registry and deploy it in Cloud Run.

Note authentication is disabled and the service account in the one configured earlier:

```sh
PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
REGION=<REPLACE_WITH_YOUR_GCP_REGION_NAME>
AR_REPO=<REPLACE_WITH_YOUR_AR_REPO_NAME>
SERVICE_NAME=genai-text-demo
gcloud artifacts repositories create $AR_REPO --location=$REGION --repository-format=Docker
gcloud auth configure-docker $REGION.pkg.dev
gcloud builds submit --tag $REGION.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --port 7860 --image $REGION.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME --service-account=cloud-run-llm@$PROJECT_ID.iam.gserviceaccount.com --allow-unauthenticated --region=$REGION --platform=managed  --project=$PROJECT_ID
```


## References

[1] PaLM-2 [technical report](https://ai.google/static/documents/palm2techreport.pdf)     
[2] YouTube video: [Generative AI on Google Cloud](https://youtu.be/Q1zF9pF6flw)      
[3] YouTube video: [Build, tune, and deploy foundation models with Vertex AI](https://youtu.be/yg2yHIKQ7oM)     
[4] YouTube video: [Build, tune, and deploy foundation models with Generative AI Support in Vertex AI](https://www.youtube.com/watch?v=-2rQ_AcQMF8)      
[5] [Overview](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) of Generative AI support on Vertex AI
