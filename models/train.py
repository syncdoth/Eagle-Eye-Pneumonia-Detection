from tqdm.auto import tqdm


def make_batch(batch):
    """Used as collate_fn in pytorch Dataloader."""
    images = []
    targets = []
    for img, targ in batch:
        images.append(img)
        targets.append(targ)

    return {"images": images, "targets": targets}


def train(model, optimizer, train_dataset, device, loss_func, val_dataset=None, epochs=1):
    """Outer training loop"""
    for epoch in range(epochs):
        model.train(True)
        train_bar = tqdm(train_dataset, desc=f'[train {epoch+1}/{epochs}]')
        for batch in train_bar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            loss = do_train_step(model, optimizer, loss_func, batch)
            # TODO: Add support for metrics, checkpointing, validation, history, etc.
            print(f"current loss: {loss}")


def do_train_step(model, optimizer, loss_func, batch, is_train=True):
    """inner training step"""
    optimizer.zero_grad()
    class_score = model(batch["images"])
    loss = loss_func(class_score, batch["targets"])
    if is_train:
        loss.backward()
        optimizer.step()
    return loss
