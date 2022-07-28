from dataset import CT_Dataset
from models.dncnn import DnCNN
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


dataset = CT_Dataset("data")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

net = DnCNN(bias=False).cuda()
optimizer = torch.optim.Adam(net.parameters())

output_dir = Path("weights")
output_dir.mkdir(exist_ok=True)

for i in range (100):
    print("-----epoch {}-----".format(i + 1))

    for (inp, tgt) in tqdm(data_loader):
        inp = inp.cuda()
        tgt = tgt.cuda()
        out = net(inp)
        loss = nn.functional.mse_loss(out, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss: {}".format(loss.item()))


torch.save(
    {"epoch": int(i), "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
    output_dir / "weights.torch"
)
