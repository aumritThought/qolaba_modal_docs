From nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN apt-get update --fix-missing
RUN apt install -y python3 python3-dev python3-pip
RUN apt install python-is-python3
RUN apt-get -y install git
RUN git clone --recursive https://github.com/qolaba/AITemplate.git
WORKDIR /AITemplate/docker
RUN bash install/install_basic_dep.sh
RUN bash install/install_test_dep.sh
RUN bash install/install_doc_dep.sh
RUN pip3 install torch torchvision torchaudio
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN bash install/install_detection_deps.sh
RUN pip3 install --upgrade diffusers==0.17.1 transformers==4.30.2 accelerate==0.20.3 scipy click ftfy
WORKDIR /AITemplate/python
RUN pwd
RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl --force-reinstall 
RUN pip3 install numpy==1.22
WORKDIR /AITemplate/examples/05_stable_diffusion/