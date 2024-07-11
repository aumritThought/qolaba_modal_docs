FROM python:3.11.8

ENV TOKEN_ID

ENV TOKEN_SECRET

RUN echo "$TOKEN_ID"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 supervisor -y

RUN apt install lsb-release curl gpg -y

RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

RUN apt-get update

RUN apt-get install redis -y

RUN mkdir /root/app

ADD ./ /root/app

WORKDIR /root/app

RUN pip install torch torchvision torchaudio && pip install -r requirements.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN modal token set --token-id $TOKEN_ID --token-secret $TOKEN_SECRET

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
