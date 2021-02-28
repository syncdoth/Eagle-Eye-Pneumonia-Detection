from tqdm.auto import tqdm
from sklearn.metrics import f1_score


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
    history = dict(train_acc=[],train_loss=[],val_acc=[],val_loss)
    
    
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        
        model.train(True)
        train_bar = tqdm(train_dataset, desc=f'[train {epoch+1}/{epochs}]')
        
        for batch in train_bar:
            
            for key in batch.keys():
                batch[key] = batch[key].to(device)
                
            loss,acc = do_train_step(model, optimizer, loss_func, batch)
            
            print(f"current loss: {loss}, current acc: {acc}",end=' /')
            train_loss.append(loss)
            train_acc.append(acc)
            # TODO: Add support for metrics, checkpointing, validation, history, etc.
        
        ### valid
        val_loss = []
        val_acc = []
        
        if val_dataset:
            model.eval()
            with torch.no_grad():
                
                for batch in val_dataset:
                    for key in batch.keys():
                        batch[key] = batch[key].to(device)

                    loss,acc = do_train_step(model, optimizer, loss_func, batch, is_train=False)
                    print(f"validation: current loss: {loss}, current f1_score: {acc}")
                    
                    val_loss.append(loss)
                    val_acc.append(acc)
                    
                    
        history['train_loss'].append(np.mean(train_loss))
        history['train_acc'].append(np.mean(train_acc))
        history['val_loss'].append(np.mean(val_loss))
        history['val_acc'].append(np.mean(val_acc))
        
            
        save_checkpoint(epoch,model,optimizer,'{}_{}_model'.format(epoch,val_loss)) # save_checkpoint
        
    
    return history        
    

def do_train_step(model, optimizer, loss_func, batch, is_train=True):
    """inner training step"""
    optimizer.zero_grad()
    class_score = model(batch["images"])
    loss = loss_func(class_score, batch["targets"])
    
#     predicted = torch.max(class_score.data, 1)[1]
#     batch_corr = (predicted == batch["targets"]).sum()
#     get accuracy
    
    f1_socre = f1_score(batch["targets"], predicted, average='weighted')
    
    if is_train:
        loss.backward()
        optimizer.step()
        
    return loss, f1_score


def save_checkpoint(epoch,model,optimizer,filename):
    state = {
    'Epoch' = epoch,
    'State_dict' = model.state_dict(),
    'optimizer' = optimizer.state_dict()
    }
    torch.save(state,filename)
    
    
