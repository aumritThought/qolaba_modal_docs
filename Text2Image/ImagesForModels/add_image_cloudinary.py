import os,io, requests, base64
from PIL import Image
url = "https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/image"
image_urls=[]
file = open('urls.txt','w')

for i in os.listdir("./"):
    list_file=[".png",".jpg"]
    if([i for j in list_file if(j in i)]):
        im=Image.open(i)
        filtered_image = io.BytesIO()
        im.save(filtered_image, "JPEG")
        myobj = {
                        "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                }
        rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
        im_url=rps.json()["data"]["secure_url"]
        image_urls.append(im_url)
        print(im_url)
        file.write(i+" :"+im_url+"\n")

file.close()