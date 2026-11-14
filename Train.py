import os, sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from Utils.getsData import get_train
from Models import get_model

def main():
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "convnext"   
    BATCH_SIZE = 32 if MODEL == "simple" else 16
    EPOCH = 30 if MODEL == "simple" else 10
    LEARNING_RATE = 1e-3 if MODEL == "simple" else 1e-4
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for fold in range(1, 6):
        train_loader, valid_loader = get_train(fold, BATCH_SIZE)
        model = get_model(MODEL).to(dev)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_loss = float("inf")
        for epoch in range(EPOCH):
            train_loss, valid_loss = 0, 0

            model.train()
            for src, trg in train_loader:
                src, trg = src.to(dev), trg.to(dev)
                pred = model(src)
                loss = loss_function(pred, trg)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for src, trg in valid_loader:
                    src, trg = src.to(dev), trg.to(dev)
                    valid_loss += loss_function(model(src), trg).item()

            loss_train_all.append(train_loss / len(train_loader))
            loss_valid_all.append(valid_loss / len(valid_loader))
            print(f"fold {fold} | epoch {epoch+1} | train loss {loss_train_all[-1]:.4f} | valid loss {loss_valid_all[-1]:.4f}")

    
            if loss_valid_all[-1] < best_loss:
                best_loss = loss_valid_all[-1]
                torch.save(model.state_dict(), f"checkpoints/{MODEL}_fold{fold}.pt")

        plt.figure()
        plt.plot(range(EPOCH), loss_train_all, color="#931a00", label="Training")
        plt.plot(range(EPOCH), loss_valid_all, color="#3399e6", label="Validation")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.title(f"{MODEL} - Fold {fold}")
        plt.savefig(f"results/{MODEL}_training_fold{fold}.png")
        plt.close()

if __name__ == "__main__":
    main()