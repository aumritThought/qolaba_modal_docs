from pydantic import constr

# Celery configuration
CELERY_RESULT_EXPIRATION_TIME = 14400
REDIS_URL = "redis://localhost:6379/0"
CELERY_MAX_RETRY = 1
CELERY_SOFT_LIMIT = 7200

#Modal app cache configuration
MAX_TIME_MODAL_APP_CACHE =  3600

# Text_to_image, Image_to_image common configuration
MIN_HEIGHT = 512
MAX_HEIGHT = 2048
MAX_INFERENCE_STEPS = 50
MIN_INFERENCE_STEPS = 5
DEFAULT_INFERENCE_STEPS = 30
MAX_GUIDANCE_SCALE = 30
MIN_GUIDANCE_SCALE = 0
DEFAULT_GUIDANCE_SCALE = 7.5
MAX_BATCH = 8
MIN_BATCH = 1
QOLABA_LORA_DIR = "Qolaba/Lora_models"
DEFAULT_LORA_SCALE = 0.5
MIN_COLOR = 0
MAX_COLOR = 255
MIN_INCREASE_SIDE = 0
MAX_INCREASE_SIDE = 2048


#Dalle configuration
DALLE_SUPPORTED_HW = ["1024x1024", "1024x1792", "1792x1024"]
DALLE_SUPPORTED_QUALITY = ["hd", "standard"]
dalle_supported_quality = constr(pattern = '|'.join(DALLE_SUPPORTED_QUALITY))


#identifiers for different app types
TEXT_TO_IMAGE_IDENTIFIER = "text2image"
IMAGE_TO_IMAGE_IDENTIFIER = "image2image"
TEXT_TO_TEXT_IDENTIFIER = "text2text"
AUDIO_IDENTIFIER = "audio"
VIDEO_IDENTIFIER = "video"
TRAINING_IDENTIFIER = "Training"

#SDXL API configuration
STABILITY_API = "https://api.stability.ai"
SDXL_ENGINE_ID = "stable-diffusion-xl-1024-v1-0"
SDXL_DEFAULT_PRESET = "enhance"


#Did api configuration
DID_TALK_API = "https://api.d-id.com/talks"
DID_AVATAR_STYLES = ["circle", "normal", "closeUp"]
did_avatar_styles = constr(pattern = '|'.join(DID_AVATAR_STYLES))
DID_EXPRESSION_LIST = ["surprise", "happy", "serious", "neutral" ]
did_expression_list = constr(pattern = '|'.join(DID_EXPRESSION_LIST))


#clipdrop api configuration
CLIPDROP_UNCROP_URL = "https://clipdrop-api.co/uncrop/v1"
CLIPDROP_CLEANUP_URL = "https://clipdrop-api.co/cleanup/v1"
CLIPDROP_REPLACE_BACKGROUND_URL = "https://clipdrop-api.co/replace-background/v1"
CLIPDROP_REMOVE_TEXT_URL = "https://clipdrop-api.co/remove-text/v1"



#Elevenlabs configuration
MAX_SUPPORTED_AUDIO_FILE_ELEVENLABS = 3
MIN_SUPPORTED_AUDIO_FILE_ELEVENLABS = 1
ELEVENLABS_GENDER_LIST = ["female","male"]
ELEVENLABS_AGE_LIST = ['young', 'middle_aged', 'old']
ELEVENLABS_ACCENT_LIST = ['british', 'american', 'african', 'australian', 'indian']
elevenlabs_age_list = constr(pattern = '|'.join(ELEVENLABS_AGE_LIST))
elevenlabs_accent_list = constr(pattern = '|'.join(ELEVENLABS_ACCENT_LIST))
elevenlabs_gender_list = constr(pattern = '|'.join(ELEVENLABS_GENDER_LIST))

#cloudinary configuration
cloudinary_supported_upload_types = [
    "image/jpeg",
    "image/png",
    "application/pdf",
    "audio/mp3",
    "audio/m4a",
    "video/mp4",
    "video/avi",
    "video/mkv",
]

#whisper configuration
whisper_supported_languages = [
    "Afrikaans",
    "Albanian",
    "Amharic",
    "Arabic",
    "Armenian",
    "Assamese",
    "Azerbaijani",
    "Bashkir",
    "Basque",
    "Belarusian",
    "Bengali",
    "Bosnian",
    "Breton",
    "Bulgarian",
    "Burmese",
    "Castilian",
    "Catalan",
    "Chinese",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Faroese",
    "Finnish",
    "Flemish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Haitian",
    "Haitian Creole",
    "Hausa",
    "Hawaiian",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Indonesian",
    "Italian",
    "Japanese",
    "Javanese",
    "Kannada",
    "Kazakh",
    "Khmer",
    "Korean",
    "Lao",
    "Latin",
    "Latvian",
    "Letzeburgesch",
    "Lingala",
    "Lithuanian",
    "Luxembourgish",
    "Macedonian",
    "Malagasy",
    "Malay",
    "Malayalam",
    "Maltese",
    "Maori",
    "Marathi",
    "Moldavian",
    "Moldovan",
    "Mongolian",
    "Myanmar",
    "Nepali",
    "Norwegian",
    "Nynorsk",
    "Occitan",
    "Panjabi",
    "Pashto",
    "Persian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Pushto",
    "Romanian",
    "Russian",
    "Sanskrit",
    "Serbian",
    "Shona",
    "Sindhi",
    "Sinhala",
    "Sinhalese",
    "Slovak",
    "Slovenian",
    "Somali",
    "Spanish",
    "Sundanese",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Tajik",
    "Tamil",
    "Tatar",
    "Telugu",
    "Thai",
    "Tibetan",
    "Turkish",
    "Turkmen",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Valencian",
    "Vietnamese",
    "Welsh",
    "Yiddish",
    "Yoruba",
]

whisper_supported_languages = constr(pattern="|".join(whisper_supported_languages))



params_range_image2image = {
    "upscaling_image2image": ["file_url", "upscale", "face_upsample"],
    "variation_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "batch",
        "num_inference_steps",
        "negative_prompt",
    ],
    "outpainting_image2image": [
        "file_url",
        "batch",
        "right",
        "left",
        "top",
        "bottom",
        "prompt",
        "negative_prompt",
    ],
    "SDXLImage2Image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "batch",
        "strength",
        "style_preset",
        "height",
        "width",
    ],
    "Background_removal_image2image": [
        "file_url",
        "bg_img",
        "bg_color",
        "r_color",
        "g_color",
        "b_color",
        "blur",
    ],
    "Face_swap_image2image": ["file_url", "bg_img"],
    "face_consistent_image2image": [
        "file_url",
        "prompt",
        "height",
        "width",
        "batch",
        "negative_prompt",
    ],
    "face_consistent_1_image2image": [
        "file_url",
        "prompt",
        "height",
        "width",
        "batch",
        "negative_prompt",
    ],
    "face_consistent_2_image2image": [
        "file_url",
        "prompt",
        "height",
        "width",
        "batch",
        "negative_prompt",
    ],
    "Illusion_Diffusion_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "QR_code_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
        "bg_img",
    ],
    "normal_controlnet__image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "scribble_controlnet__image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "seg_controlnet__image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "depth_controlnet__image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "Canny_controlnet__image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "sketch_controlnet__image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "revanim_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "3D_cartoon_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "anime_2_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "anime_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "cartoon_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "colorful_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "pixela_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "realistic_image2image": [
        "file_url",
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "realistic_2_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "ClipdropUncropImage2image": [
        "file_url",
        "right",
        "left",
        "top",
        "bottom",
        "width",
        "height",
    ],
    "ClipdropCleanupImage2image": ["file_url", "mask_url"],
    "ClipdropReplaceBackgroundImage2Image": ["file_url", "prompt"],
    "ClipdropRemoveTextImage2Image": ["file_url"],
    "Canny_controlnet_lora_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
    "kia_seltos_image2image": [
        "file_url",
        "prompt",
        "guidance_scale",
        "negative_prompt",
        "batch",
        "strength",
    ],
}

params_range_text2image = {
    "SDXLText2Image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "style_preset",
    ],
    "realistic_2_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "realistic_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "anime_2_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "anime_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "colorful_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "cartoon_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "pixela_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "3D_cartoon_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "revanim_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "Fast_SDXL_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "sdxl_turbo_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "DalleText2Image": ["height", "width", "prompt", "quality", "batch"],
    "kia_seltos_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "vr_headset_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
    ],
    "Lora_text2image": [
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "batch",
        "prompt",
        "negative_prompt",
        "lora_dir",
        "lora_model",
        "lora_scale",
    ],
}
# params_range_video={
#     "infinite_zoom_video":["file_url","List_of_prompt","zoom","Num_of_sec","height","width","guidance_scale","negative_prompt","frame_numbers","fps"],
#     "upscaling_video":["video_url","upscale","face_upsample"],
#     "Face_swap_video":["video_url","file_url"],
#     "stable_diffusion_video":["List_of_prompts_sd","Num_of_sec","height","width","guidance_scale","negative_prompt","fps"],
#     "sad_talker_video":["file_url","audio_url"],

# }

params_range_audio = {
    "whisper_audio": ["file_url", "language"],
    "AudioCraft_musicgen_audio": ["prompt", "duration"],
    "AudioCraft_audiogen_audio": ["prompt", "duration"],
    "ElvenLabsAudio": [
        "prompt",
        "clone",
        "clone_parameters",
        "voicedesign",
        "design_parameters",
        "list_of_voices",
        "voice_Data",
        "generate_audio",
        "audio_parameters",
    ],
}

params_range_text = {
    "promptparrot_text2text": [
        "prompt",
        "batch",
        "max_prompt_length",
        "min_prompt_length",
    ]
}

params_range_video = {
    "stable_diffusion_video": ["file_url"],
    "DIDVideo": ["file_url", "voice_id", "prompt", "expression", "expression_intesity"],
}

params_range_training = {
    "Lora_Training": [
        "file_url",
        "max_steps",
        "image_repetition",
        "token_string",
        "category",
        "epochs",
    ]
}


combined_params_list = {
    **params_range_text2image,
    **params_range_audio,
    **params_range_text,
    **params_range_image2image,
    **params_range_video,
    **params_range_training,
}

extra_negative_prompt = "disfigured, kitsch, ugly, oversaturated, greain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye"

SDXL_API_Style_presets = [
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

Image_HW_range = {
    "codeformer_image2image": [2048, 2048],
    "SDXL_image2image": [2048, 2048],
    "rest_models": [2048, 2048],
}


api_apps = {
    "ap-oDVXRQ7bCaWW05O52P1mlA": "SDXLText2Image",
    "ap-eRSyl0imzrUrbvgdAYwDTl": "SDXLImage2Image",
    "ap-a1Syd0inzrUrbvgdAbcser": "ElvenLabsAudio",
    "ap-sdSyd0idsndjnsnsndjsds": "DalleText2Image",
    "ap-ssndjnidsierrsnsnd223d": "DIDVideo",
    "ap-7yMwQ4XtC3ZrP2jLHJ8bB1": "ClipdropUncropImage2image",
    "ap-7XmHk4LtVgFJq2cQoE3yB8": "ClipdropCleanupImage2image",
    "ap-L4vHj7YbXeT2qKoU1fW3G7": "ClipdropReplaceBackgroundImage2Image",
    "ap-9k8P3v6zBq2s5T1rXwQm45": "ClipdropRemoveTextImage2Image",
}



# style_presets = {
#     "None": {"prompt_prefix": "", "prompt_postfix": "", "Negative_Prompt": ""},
#     "3d-model": {
#         "prompt_prefix": "professional 3d model",
#         "prompt_postfix": "octane render, highly detailed, volumetric, dramatic lighting",
#         "Negative_Prompt": "ugly, deformed, noisy, low poly, blurry, painting",
#     },
#     "analog-film": {
#         "prompt_prefix": "analog film photo",
#         "prompt_postfix": "faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
#         "Negative_Prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
#     },
#     "anime": {
#         "prompt_prefix": "anime artwork",
#         "prompt_postfix": "anime style, key visual, vibrant, studio anime, highly detailed",
#         "Negative_Prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
#     },
#     "cinematic": {
#         "prompt_prefix": "cinematic film still",
#         "prompt_postfix": "shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
#         "Negative_Prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
#     },
#     "comic-book": {
#         "prompt_prefix": "comic",
#         "prompt_postfix": "graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
#         "Negative_Prompt": "photograph, deformed, glitch, noisy, realistic, stock photo",
#     },
#     "craft clay": {
#         "prompt_prefix": "play-doh style",
#         "prompt_postfix": "sculpture, clay art, centered composition, Claymation",
#         "Negative_Prompt": "sloppy, messy, grainy, highly detailed, ultra textured, photo",
#     },
#     "digital-art": {
#         "prompt_prefix": "concept art",
#         "prompt_postfix": "digital artwork, illustrative, painterly, matte painting, highly detailed",
#         "Negative_Prompt": "photo, photorealistic, realism, ugly",
#     },
#     "enhance": {
#         "prompt_prefix": "breathtaking",
#         "prompt_postfix": "award-winning, professional, highly detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, distorted, grainy",
#     },
#     "fantasy-art": {
#         "prompt_prefix": "ethereal fantasy concept art of",
#         "prompt_postfix": "magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
#         "Negative_Prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
#     },
#     "isometric": {
#         "prompt_prefix": "isometric style",
#         "prompt_postfix": "vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
#         "Negative_Prompt": "deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic",
#     },
#     "line-art": {
#         "prompt_prefix": "line art drawing",
#         "prompt_postfix": "professional, sleek, modern, minimalist, graphic, line art, vector graphics",
#         "Negative_Prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
#     },
#     "low-poly": {
#         "prompt_prefix": "low-poly style",
#         "prompt_postfix": "low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
#         "Negative_Prompt": "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
#     },
#     "neon-punk": {
#         "prompt_prefix": "neonpunk style",
#         "prompt_postfix": "cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
#         "Negative_Prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
#     },
#     "origami": {
#         "prompt_prefix": "origami style",
#         "prompt_postfix": "paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
#         "Negative_Prompt": "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
#     },
#     "photographic": {
#         "prompt_prefix": "cinematic photo",
#         "prompt_postfix": "35mm photograph, film, bokeh, professional, 4k, highly detailed",
#         "Negative_Prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
#     },
#     "pixel-art": {
#         "prompt_prefix": "pixel-art",
#         "prompt_postfix": "low-res, blocky, pixel art style, 8-bit graphics",
#         "Negative_Prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
#     },
#     "tile-texture": {
#         "prompt_prefix": "texture",
#         "prompt_postfix": "top down close-up",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry",
#     },
#     "ads advertising": {
#         "prompt_prefix": "advertising poster style",
#         "prompt_postfix": "Professional, modern, product-focused, commercial, eye-catching, highly detailed",
#         "Negative_Prompt": "noisy, blurry, amateurish, sloppy, unattractive",
#     },
#     "ads automotive": {
#         "prompt_prefix": "automotive advertisement style",
#         "prompt_postfix": "sleek, dynamic, professional, commercial, vehicle-focused, high-resolution, highly detailed",
#         "Negative_Prompt": "noisy, blurry, unattractive, sloppy, unprofessional",
#     },
#     "ads corporate": {
#         "prompt_prefix": "corporate branding style",
#         "prompt_postfix": "professional, clean, modern, sleek, minimalist, business-oriented, highly detailed",
#         "Negative_Prompt": "noisy, blurry, grungy, sloppy, cluttered, disorganized",
#     },
#     "ads fashion editorial": {
#         "prompt_prefix": "fashion editorial style",
#         "prompt_postfix": "high fashion, trendy, stylish, editorial, magazine style, professional, highly detailed",
#         "Negative_Prompt": "outdated, blurry, noisy, unattractive, sloppy",
#     },
#     "ads food photography": {
#         "prompt_prefix": "food photography style",
#         "prompt_postfix": "appetizing, professional, culinary, high-resolution, commercial, highly detailed",
#         "Negative_Prompt": "unappetizing, sloppy, unprofessional, noisy, blurry",
#     },
#     "ads gourmet food photography": {
#         "prompt_prefix": "gourmet food photo of",
#         "prompt_postfix": "soft natural lighting, macro details, vibrant colors, fresh ingredients, glistening textures, bokeh background, styled plating, wooden tabletop, garnished, tantalizing, editorial quality",
#         "Negative_Prompt": "cartoon, anime, sketch, grayscale, dull, overexposed, cluttered, messy plate, deformed",
#     },
#     "ads luxury": {
#         "prompt_prefix": "luxury product style",
#         "prompt_postfix": "elegant, sophisticated, high-end, luxurious, professional, highly detailed",
#         "Negative_Prompt": "cheap, noisy, blurry, unattractive, amateurish",
#     },
#     "ads real estate": {
#         "prompt_prefix": "real estate photography style",
#         "prompt_postfix": "professional, inviting, well-lit, high-resolution, property-focused, commercial, highly detailed",
#         "Negative_Prompt": "dark, blurry, unappealing, noisy, unprofessional",
#     },
#     "ads retail": {
#         "prompt_prefix": "retail packaging style",
#         "prompt_postfix": "vibrant, enticing, commercial, product-focused, eye-catching, professional, highly detailed",
#         "Negative_Prompt": "noisy, blurry, amateurish, sloppy, unattractive",
#     },
#     "artstyle abstract": {
#         "prompt_prefix": "abstract style",
#         "prompt_postfix": "non-representational, colors and shapes, expression of feelings, imaginative, highly detailed",
#         "Negative_Prompt": "realistic, photographic, figurative, concrete",
#     },
#     "artstyle abstract expressionism": {
#         "prompt_prefix": "abstract expressionist painting",
#         "prompt_postfix": "energetic brushwork, bold colors, abstract forms, expressive, emotional",
#         "Negative_Prompt": "realistic, photorealistic, low contrast, plain, simple, monochrome",
#     },
#     "artstyle art deco": {
#         "prompt_prefix": "art deco style",
#         "prompt_postfix": "geometric shapes, bold colors, luxurious, elegant, decorative, symmetrical, ornate, detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, modernist, minimalist",
#     },
#     "artstyle art nouveau": {
#         "prompt_prefix": "art nouveau style",
#         "prompt_postfix": "elegant, decorative, curvilinear forms, nature-inspired, ornate, detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, modernist, minimalist",
#     },
#     "artstyle constructivist": {
#         "prompt_prefix": "constructivist style",
#         "prompt_postfix": "geometric shapes, bold colors, dynamic composition, propaganda art style",
#         "Negative_Prompt": "realistic, photorealistic, low contrast, plain, simple, abstract expressionism",
#     },
#     "artstyle cubist": {
#         "prompt_prefix": "cubist artwork",
#         "prompt_postfix": "geometric shapes, abstract, innovative, revolutionary",
#         "Negative_Prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
#     },
#     "artstyle expressionist": {
#         "prompt_prefix": "expressionist",
#         "prompt_postfix": "raw, emotional, dynamic, distortion for emotional effect, vibrant, use of unusual colors, detailed",
#         "Negative_Prompt": "realism, symmetry, quiet, calm, photo",
#     },
#     "artstyle graffiti": {
#         "prompt_prefix": "graffiti style",
#         "prompt_postfix": "street art, vibrant, urban, detailed, tag, mural",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic",
#     },
#     "artstyle hyperrealism": {
#         "prompt_prefix": "hyperrealistic art",
#         "prompt_postfix": "extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike",
#         "Negative_Prompt": "simplified, abstract, unrealistic, impressionistic, low resolution",
#     },
#     "artstyle impressionist": {
#         "prompt_prefix": "impressionist painting",
#         "prompt_postfix": "loose brushwork, vibrant color, light and shadow play, captures feeling over form",
#         "Negative_Prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
#     },
#     "artstyle pointillism": {
#         "prompt_prefix": "pointillism style",
#         "prompt_postfix": "composed entirely of small, distinct dots of color, vibrant, highly detailed",
#         "Negative_Prompt": "line drawing, smooth shading, large color fields, simplistic",
#     },
#     "artstyle pop art": {
#         "prompt_prefix": "pop Art style",
#         "prompt_postfix": "bright colors, bold outlines, popular culture themes, ironic or kitsch",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, minimalist",
#     },
#     "artstyle psychedelic": {
#         "prompt_prefix": "psychedelic style",
#         "prompt_postfix": "vibrant colors, swirling patterns, abstract forms, surreal, trippy",
#         "Negative_Prompt": "monochrome, black and white, low contrast, realistic, photorealistic, plain, simple",
#     },
#     "artstyle renaissance": {
#         "prompt_prefix": "renaissance style",
#         "prompt_postfix": "realistic, perspective, light and shadow, religious or mythological themes, highly detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, modernist, minimalist, abstract",
#     },
#     "artstyle steampunk": {
#         "prompt_prefix": "steampunk style",
#         "prompt_postfix": "antique, mechanical, brass and copper tones, gears, intricate, detailed",
#         "Negative_Prompt": "deformed, glitch, noisy, low contrast, anime, photorealistic",
#     },
#     "artstyle surrealist": {
#         "prompt_prefix": "surrealist art",
#         "prompt_postfix": "dreamlike, mysterious, provocative, symbolic, intricate, detailed",
#         "Negative_Prompt": "anime, photorealistic, realistic, deformed, glitch, noisy, low contrast",
#     },
#     "artstyle typography": {
#         "prompt_prefix": "typographic art",
#         "prompt_postfix": "stylized, intricate, detailed, artistic, text-based",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic",
#     },
#     "artstyle watercolor": {
#         "prompt_prefix": "watercolor painting",
#         "prompt_postfix": "vibrant, beautiful, painterly, detailed, textural, artistic",
#         "Negative_Prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
#     },
#     "futuristic biomechanical": {
#         "prompt_prefix": "biomechanical style",
#         "prompt_postfix": "blend of organic and mechanical elements, futuristic, cybernetic, detailed, intricate",
#         "Negative_Prompt": "natural, rustic, primitive, organic, simplistic",
#     },
#     "futuristic biomechanical cyberpunk": {
#         "prompt_prefix": "biomechanical cyberpunk",
#         "prompt_postfix": "cybernetics, human-machine fusion, dystopian, organic meets artificial, dark, intricate, highly detailed",
#         "Negative_Prompt": "natural, colorful, deformed, sketch, low contrast, watercolor",
#     },
#     "futuristic cybernetic": {
#         "prompt_prefix": "cybernetic style",
#         "prompt_postfix": "futuristic, technological, cybernetic enhancements, robotics, artificial intelligence themes",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, historical, medieval",
#     },
#     "futuristic cybernetic robot": {
#         "prompt_prefix": "cybernetic robot",
#         "prompt_postfix": "android, AI, machine, metal, wires, tech, futuristic, highly detailed",
#         "Negative_Prompt": "organic, natural, human, sketch, watercolor, low contrast",
#     },
#     "futuristic cyberpunk cityscape": {
#         "prompt_prefix": "cyberpunk cityscape",
#         "prompt_postfix": "neon lights, dark alleys, skyscrapers, futuristic, vibrant colors, high contrast, highly detailed",
#         "Negative_Prompt": "natural, rural, deformed, low contrast, black and white, sketch, watercolor",
#     },
#     "futuristic futuristic": {
#         "prompt_prefix": "futuristic style",
#         "prompt_postfix": "sleek, modern, ultramodern, high tech, detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vintage, antique",
#     },
#     "futuristic retro cyberpunk": {
#         "prompt_prefix": "retro cyberpunk",
#         "prompt_postfix": "80\u2019s inspired, synthwave, neon, vibrant, detailed, retro futurism",
#         "Negative_Prompt": "modern, desaturated, black and white, realism, low contrast",
#     },
#     "futuristic retro futurism": {
#         "prompt_prefix": "retro-futuristic",
#         "prompt_postfix": "vintage sci-fi, 50s and 60s style, atomic age, vibrant, highly detailed",
#         "Negative_Prompt": "contemporary, realistic, rustic, primitive",
#     },
#     "futuristic sci fi": {
#         "prompt_prefix": "sci-fi style",
#         "prompt_postfix": "futuristic, technological, alien worlds, space themes, advanced civilizations",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, historical, medieval",
#     },
#     "futuristic vaporwave": {
#         "prompt_prefix": "vaporwave style",
#         "prompt_postfix": "retro aesthetic, cyberpunk, vibrant, neon colors, vintage 80s and 90s style, highly detailed",
#         "Negative_Prompt": "monochrome, muted colors, realism, rustic, minimalist, dark",
#     },
#     "game bubble bobble": {
#         "prompt_prefix": "Bubble Bobble style",
#         "prompt_postfix": "8-bit, cute, pixelated, fantasy, vibrant, reminiscent of Bubble Bobble game",
#         "Negative_Prompt": "realistic, modern, photorealistic, violent, horror",
#     },
#     "game cyberpunk game": {
#         "prompt_prefix": "cyberpunk game style",
#         "prompt_postfix": "neon, dystopian, futuristic, digital, vibrant, detailed, high contrast, reminiscent of cyberpunk genre video games",
#         "Negative_Prompt": "historical, natural, rustic, low detailed",
#     },
#     "game fighting game": {
#         "prompt_prefix": "fighting game style",
#         "prompt_postfix": "dynamic, vibrant, action-packed, detailed character design, reminiscent of fighting video games",
#         "Negative_Prompt": "peaceful, calm, minimalist, photorealistic",
#     },
#     "game gta": {
#         "prompt_prefix": "GTA-style artwork",
#         "prompt_postfix": "satirical, exaggerated, pop art style, vibrant colors, iconic characters, action-packed",
#         "Negative_Prompt": "realistic, black and white, low contrast, impressionist, cubist, noisy, blurry, deformed",
#     },
#     "game mario": {
#         "prompt_prefix": "Super Mario style",
#         "prompt_postfix": "vibrant, cute, cartoony, fantasy, playful, reminiscent of Super Mario series",
#         "Negative_Prompt": "realistic, modern, horror, dystopian, violent",
#     },
#     "game minecraft": {
#         "prompt_prefix": "Minecraft style",
#         "prompt_postfix": "blocky, pixelated, vibrant colors, recognizable characters and objects, game assets",
#         "Negative_Prompt": "smooth, realistic, detailed, photorealistic, noise, blurry, deformed",
#     },
#     "game pokemon": {
#         "prompt_prefix": "Pok\u00e9mon style",
#         "prompt_postfix": "vibrant, cute, anime, fantasy, reminiscent of Pok\u00e9mon series",
#         "Negative_Prompt": "realistic, modern, horror, dystopian, violent",
#     },
#     "game retro arcade": {
#         "prompt_prefix": "retro arcade style",
#         "prompt_postfix": "8-bit, pixelated, vibrant, classic video game, old school gaming, reminiscent of 80s and 90s arcade games",
#         "Negative_Prompt": "modern, ultra-high resolution, photorealistic, 3D",
#     },
#     "game retro game": {
#         "prompt_prefix": "retro game art",
#         "prompt_postfix": "16-bit, vibrant colors, pixelated, nostalgic, charming, fun",
#         "Negative_Prompt": "realistic, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
#     },
#     "game rpg fantasy game": {
#         "prompt_prefix": "role-playing game (RPG) style fantasy",
#         "prompt_postfix": "detailed, vibrant, immersive, reminiscent of high fantasy RPG games",
#         "Negative_Prompt": "sci-fi, modern, urban, futuristic, low detailed",
#     },
#     "game strategy game": {
#         "prompt_prefix": "strategy game style",
#         "prompt_postfix": "overhead view, detailed map, units, reminiscent of real-time strategy video games",
#         "Negative_Prompt": "first-person view, modern, photorealistic",
#     },
#     "game streetfighter": {
#         "prompt_prefix": "Street Fighter style",
#         "prompt_postfix": "vibrant, dynamic, arcade, 2D fighting game, highly detailed, reminiscent of Street Fighter series",
#         "Negative_Prompt": "3D, realistic, modern, photorealistic, turn-based strategy",
#     },
#     "game zelda": {
#         "prompt_prefix": "Legend of Zelda style",
#         "prompt_postfix": "vibrant, fantasy, detailed, epic, heroic, reminiscent of The Legend of Zelda series",
#         "Negative_Prompt": "sci-fi, modern, realistic, horror",
#     },
#     "misc architectural": {
#         "prompt_prefix": "architectural style",
#         "prompt_postfix": "clean lines, geometric shapes, minimalist, modern, architectural drawing, highly detailed",
#         "Negative_Prompt": "curved lines, ornate, baroque, abstract, grunge",
#     },
#     "misc disco": {
#         "prompt_prefix": "disco-themed",
#         "prompt_postfix": "vibrant, groovy, retro 70s style, shiny disco balls, neon lights, dance floor, highly detailed",
#         "Negative_Prompt": "minimalist, rustic, monochrome, contemporary, simplistic",
#     },
#     "misc dreamscape": {
#         "prompt_prefix": "dreamscape",
#         "prompt_postfix": "surreal, ethereal, dreamy, mysterious, fantasy, highly detailed",
#         "Negative_Prompt": "realistic, concrete, ordinary, mundane",
#     },
#     "misc dystopian": {
#         "prompt_prefix": "dystopian style",
#         "prompt_postfix": "bleak, post-apocalyptic, somber, dramatic, highly detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, cheerful, optimistic, vibrant, colorful",
#     },
#     "misc fairy tale": {
#         "prompt_prefix": "fairy tale",
#         "prompt_postfix": "magical, fantastical, enchanting, storybook style, highly detailed",
#         "Negative_Prompt": "realistic, modern, ordinary, mundane",
#     },
#     "misc gothic": {
#         "prompt_prefix": "gothic style",
#         "prompt_postfix": "dark, mysterious, haunting, dramatic, ornate, detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, cheerful, optimistic",
#     },
#     "misc grunge": {
#         "prompt_prefix": "grunge style",
#         "prompt_postfix": "textured, distressed, vintage, edgy, punk rock vibe, dirty, noisy",
#         "Negative_Prompt": "smooth, clean, minimalist, sleek, modern, photorealistic",
#     },
#     "misc horror": {
#         "prompt_prefix": "horror-themed",
#         "prompt_postfix": "eerie, unsettling, dark, spooky, suspenseful, grim, highly detailed",
#         "Negative_Prompt": "cheerful, bright, vibrant, light-hearted, cute",
#     },
#     "misc kawaii": {
#         "prompt_prefix": "kawaii style",
#         "prompt_postfix": "cute, adorable, brightly colored, cheerful, anime influence, highly detailed",
#         "Negative_Prompt": "dark, scary, realistic, monochrome, abstract",
#     },
#     "misc lovecraftian": {
#         "prompt_prefix": "lovecraftian horror",
#         "prompt_postfix": "eldritch, cosmic horror, unknown, mysterious, surreal, highly detailed",
#         "Negative_Prompt": "light-hearted, mundane, familiar, simplistic, realistic",
#     },
#     "misc macabre": {
#         "prompt_prefix": "macabre style",
#         "prompt_postfix": "dark, gothic, grim, haunting, highly detailed",
#         "Negative_Prompt": "bright, cheerful, light-hearted, cartoonish, cute",
#     },
#     "misc manga": {
#         "prompt_prefix": "manga style",
#         "prompt_postfix": "vibrant, high-energy, detailed, iconic, Japanese comic style",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
#     },
#     "misc metropolis": {
#         "prompt_prefix": "metropolis-themed",
#         "prompt_postfix": "urban, cityscape, skyscrapers, modern, futuristic, highly detailed",
#         "Negative_Prompt": "rural, natural, rustic, historical, simple",
#     },
#     "misc minimalist": {
#         "prompt_prefix": "minimalist style",
#         "prompt_postfix": "simple, clean, uncluttered, modern, elegant",
#         "Negative_Prompt": "ornate, complicated, highly detailed, cluttered, disordered, messy, noisy",
#     },
#     "misc monochrome": {
#         "prompt_prefix": "monochrome",
#         "prompt_postfix": "black and white, contrast, tone, texture, detailed",
#         "Negative_Prompt": "colorful, vibrant, noisy, blurry, deformed",
#     },
#     "misc nautical": {
#         "prompt_prefix": "nautical-themed",
#         "prompt_postfix": "sea, ocean, ships, maritime, beach, marine life, highly detailed",
#         "Negative_Prompt": "landlocked, desert, mountains, urban, rustic",
#     },
#     "misc space": {
#         "prompt_prefix": "space-themed",
#         "prompt_postfix": "cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed",
#         "Negative_Prompt": "earthly, mundane, ground-based, realism",
#     },
#     "misc stained glass": {
#         "prompt_prefix": "stained glass style",
#         "prompt_postfix": "vibrant, beautiful, translucent, intricate, detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic",
#     },
#     "misc techwear fashion": {
#         "prompt_prefix": "techwear fashion",
#         "prompt_postfix": "futuristic, cyberpunk, urban, tactical, sleek, dark, highly detailed",
#         "Negative_Prompt": "vintage, rural, colorful, low contrast, realism, sketch, watercolor",
#     },
#     "misc tribal": {
#         "prompt_prefix": "tribal style",
#         "prompt_postfix": "indigenous, ethnic, traditional patterns, bold, natural colors, highly detailed",
#         "Negative_Prompt": "modern, futuristic, minimalist, pastel",
#     },
#     "misc zentangle": {
#         "prompt_prefix": "zentangle",
#         "prompt_postfix": "intricate, abstract, monochrome, patterns, meditative, highly detailed",
#         "Negative_Prompt": "colorful, representative, simplistic, large fields of color",
#     },
#     "papercraft collage": {
#         "prompt_prefix": "collage style",
#         "prompt_postfix": "mixed media, layered, textural, detailed, artistic",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic",
#     },
#     "papercraft flat papercut": {
#         "prompt_prefix": "flat papercut style",
#         "prompt_postfix": "silhouette, clean cuts, paper, sharp edges, minimalist, color block",
#         "Negative_Prompt": "3D, high detail, noise, grainy, blurry, painting, drawing, photo, disfigured",
#     },
#     "papercraft kirigami": {
#         "prompt_prefix": "kirigami representation of",
#         "prompt_postfix": "3D, paper folding, paper cutting, Japanese, intricate, symmetrical, precision, clean lines",
#         "Negative_Prompt": "painting, drawing, 2D, noisy, blurry, deformed",
#     },
#     "papercraft paper mache": {
#         "prompt_prefix": "paper mache representation of",
#         "prompt_postfix": "3D, sculptural, textured, handmade, vibrant, fun",
#         "Negative_Prompt": "2D, flat, photo, sketch, digital art, deformed, noisy, blurry",
#     },
#     "papercraft paper quilling": {
#         "prompt_prefix": "paper quilling art of",
#         "prompt_postfix": "intricate, delicate, curling, rolling, shaping, coiling, loops, 3D, dimensional, ornamental",
#         "Negative_Prompt": "photo, painting, drawing, 2D, flat, deformed, noisy, blurry",
#     },
#     "papercraft papercut collage": {
#         "prompt_prefix": "papercut collage of",
#         "prompt_postfix": "mixed media, textured paper, overlapping, asymmetrical, abstract, vibrant",
#         "Negative_Prompt": "photo, 3D, realistic, drawing, painting, high detail, disfigured",
#     },
#     "papercraft papercut shadow box": {
#         "prompt_prefix": "3D papercut shadow box of",
#         "prompt_postfix": "layered, dimensional, depth, silhouette, shadow, papercut, handmade, high contrast",
#         "Negative_Prompt": "painting, drawing, photo, 2D, flat, high detail, blurry, noisy, disfigured",
#     },
#     "papercraft stacked papercut": {
#         "prompt_prefix": "stacked papercut art of",
#         "prompt_postfix": "3D, layered, dimensional, depth, precision cut, stacked layers, papercut, high contrast",
#         "Negative_Prompt": "2D, flat, noisy, blurry, painting, drawing, photo, deformed",
#     },
#     "papercraft thick layered papercut": {
#         "prompt_prefix": "thick layered papercut art of",
#         "prompt_postfix": "deep 3D, volumetric, dimensional, depth, thick paper, high stack, heavy texture, tangible layers",
#         "Negative_Prompt": "2D, flat, thin paper, low stack, smooth texture, painting, drawing, photo, deformed",
#     },
#     "photo alien": {
#         "prompt_prefix": "alien-themed",
#         "prompt_postfix": "extraterrestrial, cosmic, otherworldly, mysterious, sci-fi, highly detailed",
#         "Negative_Prompt": "earthly, mundane, common, realistic, simple",
#     },
#     "photo film noir": {
#         "prompt_prefix": "film noir style",
#         "prompt_postfix": "monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful",
#     },
#     "photo glamour": {
#         "prompt_prefix": "glamorous photo",
#         "prompt_postfix": "high fashion, luxurious, extravagant, stylish, sensual, opulent, elegance, stunning beauty, professional, high contrast, detailed",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, distorted, grainy, sketch, low contrast, dull, plain, modest",
#     },
#     "photo hdr": {
#         "prompt_prefix": "HDR photo of",
#         "prompt_postfix": "High dynamic range, vivid, rich details, clear shadows and highlights, realistic, intense, enhanced contrast, highly detailed",
#         "Negative_Prompt": "flat, low contrast, oversaturated, underexposed, overexposed, blurred, noisy",
#     },
#     "photo iphone photographic": {
#         "prompt_prefix": "iphone photo",
#         "prompt_postfix": "large depth of field, deep depth of field, highly detailed",
#         "Negative_Prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, shallow depth of field, bokeh",
#     },
#     "photo long exposure": {
#         "prompt_prefix": "long exposure photo of",
#         "prompt_postfix": "Blurred motion, streaks of light, surreal, dreamy, ghosting effect, highly detailed",
#         "Negative_Prompt": "static, noisy, deformed, shaky, abrupt, flat, low contrast",
#     },
#     "photo neon noir": {
#         "prompt_prefix": "neon noir",
#         "prompt_postfix": "cyberpunk, dark, rainy streets, neon signs, high contrast, low light, vibrant, highly detailed",
#         "Negative_Prompt": "bright, sunny, daytime, low contrast, black and white, sketch, watercolor",
#     },
#     "photo silhouette": {
#         "prompt_prefix": "silhouette style",
#         "prompt_postfix": "high contrast, minimalistic, black and white, stark, dramatic",
#         "Negative_Prompt": "ugly, deformed, noisy, blurry, low contrast, color, realism, photorealistic",
#     },
#     "photo tilt shift": {
#         "prompt_prefix": "tilt-shift photo of",
#         "prompt_postfix": "selective focus, miniature effect, blurred background, highly detailed, vibrant, perspective control",
#         "Negative_Prompt": "blurry, noisy, deformed, flat, low contrast, unrealistic, oversaturated, underexposed",
#     },
# }



# String conditions


style_presets_list = constr(pattern = '|'.join(["enhance"]))


