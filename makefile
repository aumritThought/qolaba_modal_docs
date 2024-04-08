setup:
	python -m pip install --upgrade pip
	pip install torch torchvision torchaudio && pip install -r requirements.txt

deploy_main:
	modal run -e main src/repositories/ModalVolume.py
	modal deploy src/ModalPipelines/CustomPipelines/FRNDFaceAvatar.py --env main
	modal deploy src/ModalPipelines/BackgroundRemoval.py --env main
	modal deploy src/ModalPipelines/FaceConsistent.py --env main
	modal deploy src/ModalPipelines/IllusionDiffusion.py --env main
	modal deploy src/ModalPipelines/QrCodeGeneration.py --env main
	modal deploy src/ModalPipelines/SDXLControlnet.py --env main
	modal deploy src/ModalPipelines/SDXLImageToImage.py --env main
	modal deploy src/ModalPipelines/SDXLTextToImage.py --env main
	modal deploy src/ModalPipelines/Upscaling.py --env main
	modal deploy src/ModalPipelines/Variation.py --env main

deploy_dev:
	modal run -e dev src/repositories/ModalVolume.py
	modal deploy src/ModalPipelines/CustomPipelines/FRNDFaceAvatar.py --env dev
	modal deploy src/ModalPipelines/BackgroundRemoval.py --env dev
	modal deploy src/ModalPipelines/FaceConsistent.py --env dev
	modal deploy src/ModalPipelines/IllusionDiffusion.py --env dev
	modal deploy src/ModalPipelines/QrCodeGeneration.py --env dev
	modal deploy src/ModalPipelines/SDXLControlnet.py --env dev
	modal deploy src/ModalPipelines/SDXLImageToImage.py --env dev
	modal deploy src/ModalPipelines/SDXLTextToImage.py --env dev
	modal deploy src/ModalPipelines/Upscaling.py --env dev
	modal deploy src/ModalPipelines/Variation.py --env dev