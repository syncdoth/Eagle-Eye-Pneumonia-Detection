import torch
from tqdm.auto import tqdm


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
    class2id = {'COVID19-PN': 1, 'TheOther-PN': 2, 'Viral-PN': 3}
    for i in range(len(infer_dataset)):
        if predictions[i] == 0:
            continue
        img_name = infer_dataset.imgs[i]
        infer_dataset.data_file[infer_dataset.data_file["IMG_PATH"] ==
                                img_name]["CLASS"] = class2id[predictions[i]]

    infer_dataset.to_csv(fname, index=False)
