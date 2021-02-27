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
        # TODO: add more history categories
        loss_history = []
        f1_history = []
        val_history = []

        model.train(True)
        train_bar = tqdm(train_dataset, desc=f'[train {epoch+1}/{epochs}]')
        for batch in train_bar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            loss = do_train_step(model, optimizer, loss_func, batch)
            loss_history.append(loss)
            # TODO: Add support for metrics, checkpointing, validation, history, etc.
            print(f"current loss: {loss}")
        # validation loop
        model.eval()
        for batch in val_dataset:
            # no back prop (no model update)
            ### loss 와 metric 뽑아서 history 에 추가
            raise NotImplementedError


def do_train_step(model, optimizer, loss_func, batch, is_train=True):
    """inner training step"""
    optimizer.zero_grad()
    class_score = model(batch["images"])
    # TODO: Add metric calculation
    loss = loss_func(class_score, batch["targets"])
    if is_train:
        loss.backward()
        optimizer.step()
    return loss
