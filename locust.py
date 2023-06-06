from locust import HttpUser, task
import random
import urllib

class HelloWorldUser(HttpUser):
    host="https://qolaba--stable-diffusion-dev.modal.run"
    @task(1)

    def hello_world(self):
        url="/"
        # b=urllib.parse.quote_plus(url)
        self.client.post(url)