from pydantic import BaseModel
from uuid import uuid4
        
class Time(BaseModel):
    startup_time: str
    runtime: str

class Inference(BaseModel):
    id: uuid4 = uuid4()
    result: list
    has_nsfw_content: bool
    time: Time