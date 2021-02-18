import torch


def train(model, optimizer, train_dataset, device, loss_func, val_dataset=None, epochs=1):
    """Outer training loop"""
    for epoch in range(epochs):
        model.train(True)
        print(f"epoch number: {epoch}")
        for batch in train_dataset:
            batch["images"] = batch["images"].to(device)
            batch["labels"] = batch["labels"].to(device)
            loss = do_train_step(model, optimizer, loss_func, batch)
            # TODO: Add support for metrics, checkpointing, validation, history, etc.
            print(f"current loss: {loss}")


def do_train_step(model, optimizer, loss_func, batch, is_train=True):
    """inner training step"""
    optimizer.zero_grad()
    class_score = model(batch["images"])
    loss = loss_func(class_score, batch["labels"])
    if is_train:
        loss.backward()
        optimizer.step()
    return loss
