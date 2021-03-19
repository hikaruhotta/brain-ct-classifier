import numpy as np
from tqdm import tqdm
from torchnet import meter
import sklearn.metrics as sk_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

fn_dict_probs = {
    'AUPRC': sk_metrics.average_precision_score,
    'AUROC': sk_metrics.roc_auc_score,
    'log_loss': sk_metrics.log_loss,
}

# Functions that take binary as input
fn_dict_binary = {
    'accuracy': sk_metrics.accuracy_score,
    'precision': sk_metrics.precision_score,
    'recall': sk_metrics.recall_score,
    'f1': sk_metrics.f1_score
}

def compute_best_roc_threshold(gt, probs):
    fpr, tpr, thresholds = sk_metrics.roc_curve(gt, probs)
    J = tpr - fpr
    ix = np.argmax(J)      
    best_thresh = thresholds[ix]
    return best_thresh

def _compute_metrics(args, gts, probs, prefix):
    metrics = {}
    N = probs.shape[0]
    for i in range(N):
        gt = gts[i]
        prob = probs[i]
        
        # Determine threshold based on Precision Recall Curve
        threshold = compute_best_roc_threshold(gt, prob)
        
        # Compute binary metrics
        for eval_fn_name, eval_fn in fn_dict_binary.items():
            pred = (prob > threshold).astype(np.int32)
            try:
                score = eval_fn(gt, pred)
            except Exception as e:
                score = 0
                print(e)
            metrics[f"{prefix}_{args.features[i]}_{eval_fn_name}"] = score
                
        # Compute probability metrics
        for eval_fn_name, eval_fn in fn_dict_probs.items():
            try:
                score = eval_fn(gt, prob)
            except Exception as e:
                score = 0
                print(e)
            metrics[f"{prefix}_{args.features[i]}_{eval_fn_name}"] = score
    
    return metricds


def evaluate(args, model, loss_fn, data_loader, phase, device, prefix):
    model.eval()
    loss_meter = meter.AverageValueMeter()
    gts = []
    pbs = []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            head_preds = model.forward(inputs)
            loss = loss_fn(head_preds, targets)
            loss_meter.add(loss.item())
            gts.append(targets.cpu().detach())
            pbs.append(np.squeeze(np.array([torch.sigmoid(pred).cpu().detach().numpy() for pred in head_preds]), -1))
    
    # transpose probs and gts to get rows corresponding to each features (N, f) -> (f, N)
    pbs = np.hstack(pbs)
    gts = np.array(np.concatenate(gts).tolist()).T.astype(int)
    metrics = _compute_metrics(args, gts, pbs, prefix)
    
    # Compute average losses
    metrics[f"{prefix}_{phase}_loss"] = loss_meter.value()[0]
    
    model.train()
    
    return metrics

def evaluate_classifier(args):
    eval_dataset = ClassifierDataset(args.csv_dir, args.split, args.features, resample=(
        args.num_slices, args.slice_size, args.slice_size))
    eval_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )
    
    saver = ModelSaver(args, max_ckpts=args.max_ckpts,
                       metric_name=args.best_ckpt_metric, maximize_metric=args.maximize_metric)
    
    model = MTClassifier3D(args).to(args.device)
    saver.load_model(model, args.name, ckpt_path=args.ckpt_path)
    
    loss_wrapper = MultiTaskLoss(args)
    
    metrics = evaluate(args, model, loss_wrapper,
                           validation_loader, "validation", args.device)
    print(metrics)

    
if __name__ == "__main__":
    gts = [torch.zeros((4,7)) for i in range(3)] + [torch.zeros((1,7))]
    gts = np.array(np.concatenate(gts).tolist()).T.astype(int)
    print(gts.shape)
    
    head_preds = [np.squeeze(np.array([torch.zeros((4,1)).numpy() for i in range(7)]), -1),
                  np.squeeze(np.array([torch.zeros((4,1)).numpy() for i in range(7)]), -1),
                  np.squeeze(np.array([torch.zeros((4,1)).numpy() for i in range(7)]), -1),
                  np.squeeze(np.array([torch.zeros((1,1)).numpy() for i in range(7)]), -1)]
    for pb in head_preds:
        print(pb.shape)
    print(np.hstack(head_preds).shape)

    
    
    
