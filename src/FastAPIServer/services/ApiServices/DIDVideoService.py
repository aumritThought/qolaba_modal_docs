import io, time, os
from src.data_models.ModalAppSchemas import DIDVideoParameters
from src.utils.Globals import timing_decorator,  upload_cloudinary_image, make_request, prepare_response
from src.FastAPIServer.services.IService import IService

class DIDVideo(IService):
    def __init__(self) -> None:
        super().__init__()
        self.api_key = self.did_api_key
        self.url = self.did_url

    def check_status(self, task_id: str) -> str:
        status = None
        st = time.time()

        self.status_url = f"{self.url}/{task_id}"

        while status != "done":

            headers = {"accept": "application/json", "authorization": self.api_key}

            response = make_request(self.status_url, "GET", headers=headers)
            print(response.text)

            status = response.json()["status"]

            if status == "error":
                raise Exception(
                    response.json()["error"]["description"], "Internal Error"
                )
            if time.time() - st > 600:
                raise Exception("Request is cancelled due to timeout", "Internal Error")
            time.sleep(1)

        return response.json()["result_url"]

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : DIDVideoParameters = DIDVideoParameters(**parameters)
        payload = {
            "script": {
                "type": "text",
                "subtitles": "false",
                "provider": {"type": "elevenlabs", "voice_id": parameters.voice_id},
                "ssml": "false",
                "input": parameters.prompt,
            },
            "config": {
                "fluent": "false",
                "pad_audio": "0.0",
                "stitch": True,
                "driver_expressions": {
                    "expressions": [
                        {
                            "expression": parameters.expression,
                            "start_frame": 0,
                            "intensity": parameters.expression_intesity,
                        }
                    ]
                },
            },
            "source_url": parameters.file_url,
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": self.api_key,
        }

        response = make_request(self.url, "POST", json_data=payload, headers=headers)
        print(response.text, response.status_code)

        if response.status_code != 201:
            print(response.json())
            raise Exception(response.json()["description"], "Internal Error")

        task_id = response.json()["id"]
        vid_url = self.check_status(task_id)

        response = make_request(vid_url, "GET")

        video_bytes = io.BytesIO(response.content)

        cld_vid_url = upload_cloudinary_image(video_bytes)
        
        return prepare_response([cld_vid_url], [False], 0, 0)

