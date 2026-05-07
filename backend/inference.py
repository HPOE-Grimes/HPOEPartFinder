import json
from io import BytesIO


class PartClassifier:
    def __init__(self, model_path: str, class_names_path: str):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image

        with open(class_names_path) as f:
            self.class_names = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self._torch = torch
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._Image = Image

    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[dict]:
        torch = self._torch
        img = self._Image.open(BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = probs.topk(min(top_k, len(self.class_names)))
        return [
            {'part_name': self.class_names[idx.item()], 'confidence': round(prob.item(), 3)}
            for prob, idx in zip(top_probs, top_indices)
        ]
