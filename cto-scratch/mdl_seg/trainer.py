import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from .dataset import ISICDataset
from .losses import CTOLoss
from .metrics import dice_iou
from .network import build_model
from .options import run_dir, save_options
from .util import AverageMeter, count_parameters, poly_lr, save_checkpoint, set_seed


class CTOTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.dir = run_dir(opt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = opt.amp and self.device.type == "cuda"
        self.start_epoch = 0
        self.best_dice = -1.0

    # ----- préparation -----
    def set_network(self):
        self.net = build_model(self.opt).to(self.device)
        print(f"=> {self.opt.model} : {count_parameters(self.net) / 1e6:.2f} M de paramètres")

    def set_loss(self):
        self.criterion = CTOLoss(
            num_class=self.opt.num_class,
            alpha=self.opt.alpha,
            edge_loss=self.opt.edge_loss,
            edge_width=self.opt.edge_width,
            boundary_weight=self.opt.boundary_weight,
        )

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

    def set_dataloader(self):
        opt = self.opt
        train_set = ISICDataset(opt.data_dir, "train", opt.fold)
        val_set = ISICDataset(opt.data_dir, "val", opt.fold)
        common = dict(num_workers=opt.workers, pin_memory=self.device.type == "cuda")
        self.train_loader = DataLoader(train_set, opt.batch_size, shuffle=True, drop_last=True, **common)
        self.val_loader = DataLoader(val_set, opt.batch_size, shuffle=False, **common)
        print(f"=> {len(train_set)} images d'entraînement, {len(val_set)} de validation")

    def resume(self):
        path = os.path.join(self.dir, "last.pth")
        if not (self.opt.resume and os.path.exists(path)):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])  # strict : une clé manquante lève une erreur
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"]
        self.best_dice = ckpt["best_dice"]
        print(f"=> reprise à l'époque {self.start_epoch} (meilleur Dice {self.best_dice:.4f})")

    # ----- une époque -----
    def train_epoch(self):
        self.net.train()
        meters = {}
        for batch in self.train_loader:
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                seg_outputs, edge_logits = self.net(image)
                loss, parts = self.criterion(seg_outputs, edge_logits, label)

            self.scaler.scale(loss).backward()
            if self.opt.clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.opt.clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            parts["loss"] = loss.item()
            for key, value in parts.items():
                meters.setdefault(key, AverageMeter()).update(value, image.size(0))
        return {key: meter.avg for key, meter in meters.items()}

    @torch.no_grad()
    def validate(self):
        self.net.eval()
        dice_meter, iou_meter = AverageMeter(), AverageMeter()
        for batch in self.val_loader:
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                seg_outputs, _ = self.net(image)
            pred = seg_outputs[0].argmax(dim=1)  # sortie la plus fine
            dice, iou = dice_iou(pred, label, self.opt.num_class)
            dice_meter.update(dice.mean().item(), image.size(0))
            iou_meter.update(iou.mean().item(), image.size(0))
        return dice_meter.avg, iou_meter.avg

    # ----- boucle complète -----
    def run(self):
        opt = self.opt
        set_seed(opt.seed)
        save_options(opt)
        self.set_network()
        self.set_loss()
        self.set_optimizer()
        self.set_dataloader()
        self.resume()

        log_path = os.path.join(self.dir, "log.csv")
        if self.start_epoch == 0:
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "lr", "loss", "loss_seg", "loss_edge", "val_dice", "val_iou"])

        for epoch in range(self.start_epoch, opt.epochs):
            t0 = time.time()
            lr = poly_lr(self.optimizer, opt.lr, epoch, opt.epochs)
            train = self.train_epoch()
            val_dice, val_iou = self.validate()

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, f"{lr:.3e}", f"{train['loss']:.4f}", f"{train['seg']:.4f}",
                    f"{train.get('edge', float('nan')):.4f}", f"{val_dice:.4f}", f"{val_iou:.4f}",
                ])

            is_best = val_dice > self.best_dice
            self.best_dice = max(self.best_dice, val_dice)
            state = {
                "epoch": epoch + 1,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_dice": self.best_dice,
            }
            save_checkpoint(state, os.path.join(self.dir, "last.pth"))
            if is_best:
                save_checkpoint(state, os.path.join(self.dir, "best.pth"))

            print(
                f"époque {epoch + 1:3d}/{opt.epochs}  loss {train['loss']:.4f}  "
                f"val Dice {val_dice:.4f}  val IoU {val_iou:.4f}  "
                f"{'(meilleur) ' if is_best else ''}{time.time() - t0:.0f}s"
            )
        print(f"=> entraînement terminé, meilleur Dice de validation : {self.best_dice:.4f}")
