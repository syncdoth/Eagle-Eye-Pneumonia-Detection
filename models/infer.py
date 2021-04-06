import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

def infer(model, infer_dataset, device):
    """inference for unclassified-PN data"""
    y_pred = []
    model.eval()
    with torch.no_grad():
        infer_bar = tqdm(infer_dataset, desc="[inference]")
        for batch in infer_bar:
            batch = batch.to(device)
            class_score = model(batch)
            pred = torch.max(class_score.data, 1)[1]
            y_pred.extend(pred.tolist())

    print("inference complete")
    return y_pred


def write_to_file(predictions, infer_dataset, fname):
    class2id = {0: 'TheOther-PN', 1: 'Viral-PN'}
    for i in range(len(infer_dataset)):
        img_name = infer_dataset.imgs[i]
        infer_dataset.data_file[infer_dataset.data_file["IMG_PATH"] ==
                                img_name]["CLASS"] = class2id[predictions[i]]
    
    infer_dataset.data_file.drop("IMG_PATH")
    infer_dataset.data_file.to_csv(fname, index=False)


def evaluate(model,
             eval_dataset,
             device):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(eval_dataset)
        for batch in val_bar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            class_score = model(batch["images"])
            predicted = torch.max(class_score.data, 1)[1]

            y_pred.extend(predicted.tolist())
            y_true.extend(batch["targets"].tolist())

    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    return y_pred