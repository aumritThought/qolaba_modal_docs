FROM python:3.11.8

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 supervisor -y

RUN apt install lsb-release curl gpg -y

RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

RUN apt-get update

RUN apt-get install redis -y

RUN mkdir /root/app

ADD ./ /root/app

WORKDIR /root/app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r req_docker_deploy.txt

ARG TOKEN_ID

ARG TOKEN_SECRET

# Define build-time arguments
ARG API_KEY
ARG BUCKET_NAME
ARG CLAUDE_API_KEY
ARG DID_KEY
ARG ELEVENLABS_API_KEY
ARG GCP_AUTH_PROVIDER_X509_CERT_URL
ARG GCP_AUTH_URI
ARG GCP_CLIENT_EMAIL
ARG GCP_CLIENT_ID
ARG GCP_CLIENT_X509_CERT_URL
ARG GCP_PRIVATE_KEY
ARG GCP_PRIVATE_KEY_ID
ARG GCP_PROJECT_ID
ARG GCP_TOKEN_URI
ARG GCP_TYPE
ARG GCP_UNIVERSE_DOMAIN
ARG NUM_WORKERS
ARG OPENAI_API_KEY
ARG QOLABA_B2B_API_URL
ARG SDXL_API_KEY
ARG ENVIRONMENT
ARG CDN_API
ARG REPLICATE_API_TOKEN
# Set environment variables
ENV API_KEY=${API_KEY}
ENV BUCKET_NAME=${BUCKET_NAME}
ENV CLAUDE_API_KEY=${CLAUDE_API_KEY}
ENV DID_KEY=${DID_KEY}
ENV ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
ENV GCP_AUTH_PROVIDER_X509_CERT_URL=${GCP_AUTH_PROVIDER_X509_CERT_URL}
ENV GCP_AUTH_URI=${GCP_AUTH_URI}
ENV GCP_CLIENT_EMAIL=${GCP_CLIENT_EMAIL}
ENV GCP_CLIENT_ID=${GCP_CLIENT_ID}
ENV GCP_CLIENT_X509_CERT_URL=${GCP_CLIENT_X509_CERT_URL}
ENV GCP_PRIVATE_KEY=${GCP_PRIVATE_KEY}
ENV GCP_PRIVATE_KEY_ID=${GCP_PRIVATE_KEY_ID}
ENV GCP_PROJECT_ID=${GCP_PROJECT_ID}
ENV GCP_TOKEN_URI=${GCP_TOKEN_URI}
ENV GCP_TYPE=${GCP_TYPE}
ENV GCP_UNIVERSE_DOMAIN=${GCP_UNIVERSE_DOMAIN}
ENV NUM_WORKERS=${NUM_WORKERS}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV QOLABA_B2B_API_URL=${QOLABA_B2B_API_URL}
ENV SDXL_API_KEY=${SDXL_API_KEY}
ENV environment=${ENVIRONMENT}
ENV CDN_API=${CDN_API}
ENV TOKEN_ID=${TOKEN_ID}
ENV TOKEN_SECRET=${TOKEN_SECRET}
ENV REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}

# Rest of your Dockerfile instructions
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN modal token set --token-id $TOKEN_ID --token-secret $TOKEN_SECRET

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]