import torch as th
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import pandas as pd


def main():
    t_loss=[]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)

    device = th.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("../../../content/drive/MyDrive/dataset/model_ckpt_5", config=config)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=4, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")
            t_loss.append(loss.item())
        losss=pd.DataFrame({'train_loss':t_loss})
        losss.to_csv(f'./epoch{epoch}_loss.csv')
        t_loss=[]
        if epoch==5:
            model.save_pretrained(f'model_ckpts_{epoch}/')
    model.save_pretrained("model_ckpts_{epoch}/")
if __name__ == "__main__":
    main()
