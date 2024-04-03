import torch
from torchvision.transforms import v2

H, W = 28, 28
img = torch.randint(0, 256, size=(1, H, W), dtype=torch.uint8)

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(0.0118, 0.0036),
])
img = transforms(img)


    # def getitem(self, idx):

    #     image = self.data.iloc[idx, 1:].to_numpy().reshape(1, 28, 28)
    #     label = self.data.iloc[idx]['label']

    #     image = image.astype('uint8') 
    #     t = T.Compose([T.ToTensor(), T.Normalize(0.0118, 0.0036)])
    #     image = t(image)
    #     image = torch.permute(image, (1, 0, 2)).float()

    #     return image, torch.tensor(label)
