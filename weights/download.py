#!pip install torch torchvision transformers
import os
import torch
from transformers import AutoModel
import torchvision.models as models

# Tạo thư mục weights nếu chưa có
os.makedirs("weights", exist_ok=True)

print("Đang tải ClinicalBERT...")
# Tải ClinicalBERT từ Hugging Face
clinical_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# Lưu state_dict
torch.save(clinical_bert.state_dict(), "weights/clinicalbert_state_dict.pt")
print("Đã lưu weights/clinicalbert_state_dict.pt")

print("Đang tải ResNet50...")
# Tải ResNet50 (ImageNet Pre-trained) từ PyTorch Hub
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Lưu state_dict
torch.save(resnet50.state_dict(), "weights/resnet50_vision_state.pt")
print("Đã lưu weights/resnet50_vision_state.pt")

print("Hoàn tất!")
