import arvcNN.Config
import arvcNN.Trainer
from arvcNN.Datasets import RandDataset
from torch.utils.data import DataLoader
import torch

config = arvcNN.Config.Config('/home/fran/PycharmProjects/arvc_Utils/config.yaml')

trainer = arvcNN.Trainer.Trainer(config_obj_=config)


dataset = RandDataset()
dataloader = DataLoader(dataset=dataset,
                        batch_size=trainer.config.train.batch_size,
                        num_workers=10,
                        pin_memory=True,
                        shuffle=True,
                        drop_last=False)


trainer.dataloader = dataloader
trainer.train()

# for batch, data in enumerate(dataloader):

#     print(f'Len of data: {len(data)}')            
#     features = data[0].to(trainer.device, dtype=torch.float32)
#     labels = data[1].to(trainer.device, dtype=torch.int32)

#     # if len(data) == 3:
#     #     filenames = data[2]  



