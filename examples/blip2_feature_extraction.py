import sys
import os.path as osp
print(sys.path)
sys.path.append(osp.join(sys.path[0], '..'))

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt
from PIL import Image

# 打开并显示图像
raw_image = Image.open("./docs/_static/merlion.png").convert("RGB")

plt.imshow(raw_image)
plt.axis('off')  # 不显示坐标轴
# plt.show()


caption = "a large fountain spewing water into the air"

# raw_image.resize((596, 437)).show()
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](caption)
sample = {"image": image, "text_input": [text_input]}
features_multimodal = model.extract_features(sample)
features_text = model.extract_features(sample, mode="text")
print(features_multimodal.multimodal_embeds.shape)
# torch.Size([1, 32, 768]), 32 is the number of queries

features_image = model.extract_features(sample, mode="image")
features_text = model.extract_features(sample, mode="text")
print(features_image.image_embeds.shape)
# torch.Size([1, 32, 768])
print(features_text.text_embeds.shape)
# torch.Size([1, 12, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 32, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 12, 256])
similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,:,:].transpose(1, 2)).max()
print(similarity)
# tensor([[0.3642]])