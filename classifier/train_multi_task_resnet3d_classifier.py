from torchnet import meter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from args.multi_task_classifier_3d_train_arg_parser import MTClassifier3DTrainArgParser
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver
from classifier.multi_task_classifier_3d import MTClassifier3D
from classifier.multi_task_loss import MultiTaskLoss
from classifier.scheduler import WarmupMultiStepLR
from classifier.evaluate_multi_task_classifier_3d import evaluate
from datasets.classifier_dataset import ClassifierDataset


def train(args):
    """
    Implements the training loop for the MultiTaskResnet3dClassifier.
    Args:
        args (Namespace) : Program arguments
    """
    # Get model and loss function
    model = MTClassifier3D(args).to(args.device)

    # Initialize losses for each head
    loss_wrapper = MultiTaskLoss(args)
    loss_fn = nn.BCEWithLogitsLoss()

    # TODO: Get train and validation dataloaders
    train_dataset = ClassifierDataset(args.csv_dir, 'train', args.features, resample=(
        args.num_slices, args.slice_size, args.slice_size))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    
    peds_validation_dataset = ClassifierDataset(args.peds_csv_dir, 'val', args.peds_features, resample=(
        args.num_slices, args.slice_size, args.slice_size))
    peds_validation_loader = DataLoader(
        peds_validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )
    
    adult_validation_dataset = ClassifierDataset(args.adult_csv_dir, 'val', args.adult_features, resample=(
        args.num_slices, args.slice_size, args.slice_size))
    adult_validation_loader = DataLoader(
        adult_validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr)
    warmup_iters = args.lr_warmup_epochs * len(train_loader)
    lr_milestones = [len(train_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    # Get saver, logger, and evaluator
    saver = ModelSaver(args, max_ckpts=args.max_ckpts,
                       metric_name=args.best_ckpt_metric, maximize_metric=args.maximize_metric)
    # evaluator = ModelEvaluator(args, validation_loader, cls_loss_fn)

    # Load model from checkpoint is applicable
    if args.continue_train:
        saver.load_model(model, args.name, ckpt_path=args.load_path,
                         optimizer=optimizer, scheduler=lr_scheduler)
    logger = TrainLogger(args, len(train_loader.dataset))


    # Multi GPU training if applicable
    if len(args.gpu_ids) > 1:
        print("Using", len(args.gpu_ids), "GPUs.")
        model = nn.DataParallel(model)

    loss_meter = meter.AverageValueMeter()

    # Train model
    logger.log_hparams(args)
    while not logger.is_finished_training():
        logger.start_epoch()

        for inputs, targets in tqdm(train_loader):
            logger.start_iter()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                head_preds = model(inputs)

                loss = loss_wrapper(head_preds, targets)
                loss_meter.add(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Log all train losses
            if logger.iter % args.steps_per_print == 0 and logger.iter != 0:
                logger.log_metrics({'train_loss': loss_meter.value()[0]})
                loss_meter.reset()

            logger.end_iter()

        # Evaluate model and save model ckpt
        if logger.epoch % args.epochs_per_eval == 0:
            peds_metrics = evaluate(args, model, loss_wrapper,
                            peds_validation_loader, "validation", args.device, 'peds')
            logger.log_metrics(peds_metrics)
            adult_metrics = evaluate(args, model, loss_wrapper,
                            adult_validation_loader, "validation", args.device, 'adult')
            logger.log_metrics(adult_metrics)
        
        if logger.epoch % args.epochs_per_save == 0:
            saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
                       args.name)
        lr_scheduler.step()
        logger.end_epoch()


if __name__ == "__main__":
    parser = MTClassifier3DTrainArgParser()
    args = parser.parse_args()
    if args.train_age_group == "peds":
        args.csv_dir = args.peds_csv_dir
        args.features = args.peds_features
    else:
        args.csv_dir = args.adult_csv_dir
        args.features = args.adult_features
    print(args)
    train(args)
