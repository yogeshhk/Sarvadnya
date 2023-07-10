## Build and deploy in Cloud Run

To deploy the Flask Application in [Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container), you need to build the Docker image in Artifact Registry and deploy it in Cloud Run.

First step is to add your Google Project ID in the `app.py` file. 

Next, look at the following script, replace the variables at the start and run the commands one after the other. This assumes that you have `gcloud` setup on your machine. 

```sh
PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
REGION=<REPLACE_WITH_YOUR_GCP_REGION_NAME>
AR_REPO=<REPLACE_WITH_YOUR_AR_REPO_NAME>
SERVICE_NAME=flask-hello-world-chat
gcloud artifacts repositories create $AR_REPO --location=$REGION --repository-format=Docker
gcloud auth configure-docker $REGION.pkg.dev
gcloud builds submit --tag $REGION.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --port 8080 --image $REGION.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME --allow-unauthenticated --region=$REGION --platform=managed  --project=$PROJECT_ID
```