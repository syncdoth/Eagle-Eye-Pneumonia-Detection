import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import os

def make_batch(batch):
    """Used as collate_fn in pytorch Dataloader."""
    images = []
    targets = []
    for img, targ in batch:
        images.append(img)
        targets.append(targ)

    return {"images": torch.stack(images, 0), "targets": torch.cat(targets)}


def train(model, optimizer, train_dataset, device, loss_func, val_dataset=None, epochs=1):
    """Outer training loop"""
    history = dict(train_acc=[], train_loss=[], val_acc=[], val_f1=[], val_loss=[])
    curr_best_val_acc = 0
    
    for epoch in range(epochs):

        model.train(True)
        train_bar = tqdm(train_dataset, desc=f"[train {epoch + 1}/{epochs}]")
        
        correct_num = 0
        total_num = 0
        running_loss = 0        

        for batch in train_bar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            pred, loss = do_train_step(model, optimizer, loss_func, batch)
            running_loss += loss.item()
            correct_num += (pred == batch['targets']).sum().item()
            total_num += batch['targets'].shape[0]
        
        train_loss = running_loss / total_num
        train_acc = correct_num / total_num
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(
            f"[train] epoch: {epoch}, loss: {train_loss},"
            f" accracy: {train_acc}")
        

        
        ### valid

        if val_dataset is None:
            save_checkpoint(epoch, model, optimizer,
                            "{}_{}_model".format(epoch, train_loss))  #save_checkpoint
            continue
            
        val_running_loss = 0
        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for batch in val_dataset:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                pred, loss = do_train_step(model,
                                         optimizer,
                                         loss_func,
                                         batch,
                                         is_train=False)
                val_running_loss += loss.item()
                y_pred.extend(pred.tolist())
                y_true.extend(batch['targets'].tolist())
                
        report = classification_report(y_true, y_pred, output_dict=True) 
        val_loss = val_running_loss / len(val_dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(report['accuracy'])
        history["val_f1"].append(report['macro avg']['f1-score'])
        print(
            f"[valid] epoch: {epoch}, loss: {val_loss},"
            f" report:\n{report}")
        if report["accuracy"] > curr_best_val_acc:
            curr_best_val_acc = report["accuracy"]
            save_checkpoint(epoch, model, optimizer,
                            "acc_{}_model".format(curr_best_val_acc))  # save_checkpoint

    return history

def do_train_step(model, optimizer, loss_func, batch, is_train=True):
    """inner training step"""
    optimizer.zero_grad()
    class_score = model(batch["images"])
    loss = loss_func(class_score, batch["targets"])

    predicted = torch.max(class_score.data, 1)[1]
    

    if is_train:
        loss.backward()
        optimizer.step()

    return predicted, loss


def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        "Epoch": epoch,
        "State_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, filename)
