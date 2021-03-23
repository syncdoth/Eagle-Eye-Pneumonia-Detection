import os
import pickle

import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report


def make_batch(batch):
    """Used as collate_fn in pytorch Dataloader."""
    images = []
    targets = []
    for img, targ in batch:
        images.append(img)
        targets.append(targ)

    return {"images": torch.stack(images, 0), "targets": torch.cat(targets)}


def train(model,
          optimizer,
          train_dataset,
          device,
          loss_func,
          val_dataset=None,
          epochs=1,
          model_dir=None):
    """Outer training loop"""
    history = dict(train_acc=[], train_loss=[], val_acc=[], val_f1=[], val_loss=[])
    curr_best_val_acc = 0

    if os.path.exists(model_dir):
        model_name = os.path.join(model_dir, "model.pth")
        history_name = os.path.join(model_dir, "hist.pickle")

        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["State_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resuming_epoch = checkpoint["Epoch"]
        print(f"resuming training from epoch {resuming_epoch + 1}...")
        with open(history_name, "rb") as handle:
            history = pickle.load(handle)
    else:
        os.makedirs(model_dir, exist_ok=True)

    for epoch in range(resuming_epoch + 1, epochs):
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
            correct_num += (pred == batch["targets"]).sum().item()
            total_num += batch["targets"].shape[0]

            train_loss = running_loss / total_num
            train_acc = correct_num / total_num
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            train_bar.set_postfix(loss=train_loss, acc=train_acc)
        print(f"[train] epoch: {epoch}, loss: {train_loss}," f" accracy: {train_acc}")

        ### valid
        if val_dataset is None:
            save_checkpoint(epoch, model, optimizer, model_name)  #save_checkpoint
            save_history(history, history_name)
            continue

        val_running_loss = 0
        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_dataset, desc=f"[val {epoch + 1}/{epochs}]")
            for batch in val_bar:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                pred, loss = do_train_step(model,
                                           optimizer,
                                           loss_func,
                                           batch,
                                           is_train=False)
                val_running_loss += loss.item()
                y_pred.extend(pred.tolist())
                y_true.extend(batch["targets"].tolist())

        report = classification_report(y_true, y_pred, output_dict=True)
        val_loss = val_running_loss / len(val_dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(report["accuracy"])
        history["val_f1"].append(report["macro avg"]["f1-score"])
        print(f"[valid] epoch: {epoch}, loss: {val_loss},"
              f" report:\n{classification_report(y_true, y_pred)}")
        if report["accuracy"] > curr_best_val_acc:
            curr_best_val_acc = report["accuracy"]
            save_checkpoint(epoch, model, optimizer, model_name)  # save_checkpoint
        # save history after every epoch
        save_history(history, history_name)

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
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, filename)


def save_history(history, save_path):
    with open(save_path, "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
