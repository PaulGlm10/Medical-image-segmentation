import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ISICDataset
from .metrics import dice_iou
from .network import build_model
from .util import count_flops, count_parameters


class CTOInference:
    def __init__(self, opt, checkpoint="best", tta=False, save_pred=False, external_dir=None, external_name=None):
        """external_dir : dossier d'un autre jeu de données (ex. PH2). Le modèle y est évalué
        sur toutes les images, sans réentraînement, et les résultats vont dans un sous-dossier."""
        self.opt = opt
        self.tta = tta
        self.save_pred = save_pred
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run = os.path.join(opt.save_dir, opt.name, f"fold_{opt.fold}")
        self.ckpt_path = os.path.join(run, f"{checkpoint}.pth")
        self.data_dir, self.phase, self.dir = opt.data_dir, "test", run
        if external_dir:
            name = external_name or os.path.basename(os.path.normpath(external_dir))
            self.data_dir, self.phase = external_dir, "all"
            self.dir = os.path.join(run, f"external_{name}")
            os.makedirs(self.dir, exist_ok=True)

    def set_network(self):
        self.opt.pretrained = False  # inutile de retélécharger ImageNet : on charge nos poids
        self.net = build_model(self.opt).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])  # strict
        self.net.eval()
        print(f"=> poids chargés : {self.ckpt_path} (époque {ckpt['epoch']})")

    @torch.no_grad()
    def predict(self, image):
        """Probabilités (B, C, H, W). Avec tta=True, moyenne sur les retournements."""
        flips = [(), (3,), (2,), (2, 3)] if self.tta else [()]
        prob = 0
        for dims in flips:
            x = torch.flip(image, dims) if dims else image
            p = torch.softmax(self.net(x)[0][0].float(), dim=1)
            prob = prob + (torch.flip(p, dims) if dims else p)
        return prob / len(flips)

    @torch.no_grad()
    def run(self):
        opt = self.opt
        self.set_network()
        loader = DataLoader(ISICDataset(self.data_dir, self.phase, opt.fold), opt.batch_size, num_workers=opt.workers)
        pred_dir = os.path.join(self.dir, "predictions")
        if self.save_pred:
            os.makedirs(pred_dir, exist_ok=True)

        rows = []
        for batch in loader:
            image, label = batch["image"].to(self.device), batch["label"].to(self.device)
            pred = self.predict(image).argmax(dim=1)
            dice, iou = dice_iou(pred, label, opt.num_class)
            for name, d, i, p in zip(batch["name"], dice.tolist(), iou.tolist(), pred.cpu().numpy()):
                rows.append([name, d, i])
                if self.save_pred:
                    import cv2
                    cv2.imwrite(os.path.join(pred_dir, f"{name}_pred.png"), (p * 255 // max(opt.num_class - 1, 1)).astype(np.uint8))

        mean_dice = float(np.mean([r[1] for r in rows]))
        mean_iou = float(np.mean([r[2] for r in rows]))
        params = count_parameters(self.net) / 1e6
        flops = count_flops(self.net, device=self.device)

        with open(os.path.join(self.dir, "test_results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "dice", "iou"])
            writer.writerows(rows)
            writer.writerow(["MOYENNE", mean_dice, mean_iou])

        summary = {"dice": mean_dice, "iou": mean_iou, "params_M": params, "flops_G": flops, "n_images": len(rows)}
        with open(os.path.join(self.dir, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        flops_txt = f"{flops:.2f} G" if flops is not None else "non mesurés (pip install thop)"
        print(f"=> {len(rows)} images de test ({self.data_dir})")
        print(f"   Dice {100 * mean_dice:.2f}   IoU {100 * mean_iou:.2f}")
        print(f"   paramètres {params:.2f} M   FLOPs {flops_txt}")
        return summary
