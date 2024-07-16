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

RUN pip install -r req_docker_deploy.txt

ARG TOKEN_ID

ARG TOKEN_SECRET

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN modal token set --token-id $TOKEN_ID --token-secret $TOKEN_SECRET

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
