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
    history = dict(train_f1=[],train_loss=[],val_f1=[],val_loss=[])
    
    
    for epoch in range(epochs):
        train_loss = []
        train_f1 = []
        
        model.train(True)
        train_bar = tqdm(train_dataset, desc=f'[train {epoch+1}/{epochs}]')
        
        for batch in train_bar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
                
            loss, f1 = do_train_step(model, optimizer, loss_func, batch)
            
            print(f"current loss: {loss}, current f1_score: {f1}",end=' /')
            train_loss.append(loss)
            train_f1.append(f1)
        
        train_loss = np.mean(train_loss)
        train_f1 = np.mean(train_f1)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        
        ### valid
       
        if val_dataset is None:
            save_checkpoint(epoch,model,optimizer,'{}_{}_model'.format(epoch,train_loss))#save_checkpoint
            continue
            
        val_loss = []
        val_f1 = []
        
        model.eval()
        with torch.no_grad():
            for batch in val_dataset:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                loss, f1 = do_train_step(model, optimizer, loss_func, batch, is_train=False)
                print(f"validation: current loss: {loss}, current f1_score: {f1}")

                val_loss.append(loss)
                val_f1.append(f1)
                    
        val_loss = np.mean(val_loss)
        val_f1 = np.mean(val_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
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
    
    f1_socre = f1_score(batch["targets"], predicted, average='macro')
    
    if is_train:
        loss.backward()
        optimizer.step()
        
    return loss, f1_score


def save_checkpoint(epoch, model, optimizer, filename):
    state = {
    'Epoch' = epoch,
    'State_dict' = model.state_dict(),
    'optimizer' = optimizer.state_dict()
    }
    torch.save(state,filename)
    
    
