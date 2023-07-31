import os,io, requests, base64
from PIL import Image


# for i in os.listdir("./"):
#     list_file=[".png",".jpg"]
#     if([i for j in list_file if(j in i)]):
#         im=Image.open(i)
#         filtered_image = io.BytesIO()
#         im.save(filtered_image, "JPEG")
#         myobj = {
#                         "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
#                 }
#         rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
#         im_url=rps.json()["data"]["secure_url"]
#         image_urls.append(im_url)
#         print(im_url)
#         file.write(i+" :"+im_url+"\n")

# file.close()
files=os.listdir("./")
print(len(files))
for i in range(0, len(files)):
    if((".py" in files[i])):
        if(not(files[i]=="deployatonce.py")):
            if(not(files[i]=="common_code.py")):
                print(files[i])
                os.system("modal deploy "+files[i])
