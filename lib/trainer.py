import sys
import warnings

import torch
import torch.distributed as dist
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from .dataset import get_test_dataset, get_train_dataloader, get_val_dataloader
from .distributed import DDP
from .image_utils import quantize, save_image
from .lr_sched import LRScheduler
from .metrics import LPIPS, gan_loss, psnr, ssim
from .model import make_model
from .model.discriminator import VGGDiscriminator
from .model.percep import VGGPercepLoss
from .utils import LazyValues, sample_to_device


def get_number_of_parameters(model):
    return sum(x.numel() for x in model.parameters())


class DummyWriter:
    def __getattr__(self, attr):
        return lambda *args: None


class Trainer:
    def __init__(self, options):
        self.options = options

        if self.options.train.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if self.options.train.visualize_batch_size % self.world_size != 0:
            warnings.warn(
                "visualize batch size is not divisible by world size, batch size will be inaccurate"
            )
        self.visualize_batch_size_per_node = (
            self.options.train.visualize_batch_size // self.world_size
        )

        if self.options.train.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = make_model(self.options)
        if self.options.train.finetune_pretrained_model is not None:
            checkpoint = torch.load(
                self.options.train.finetune_pretrained_model,
                map_location=torch.device("cpu"),
            )
            if "model_avg" in checkpoint:
                self.model.load_state_dict(checkpoint["model_avg"])
            else:
                self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
        if self.options.train.finetune_gan:
            self.discriminator = VGGDiscriminator(self.options)
            self.discriminator = self.discriminator.to(self.device)

        self.model.set_image_size(
            (self.options.train.train_image_size, self.options.train.train_image_size)
        )

        if self.options.model.type == "ecdp":
            self.optim_model = optim.Adam(
                [
                    {"params": self.model.model.diffusion.parameters()},
                    {"params": self.model.model.lr_feats.parameters(), "lr": 1e-5},
                ],
                lr=self.options.train.lr,
            )
        else:
            self.optim_model = optim.Adam(
                self.model.parameters(), lr=self.options.train.lr
            )
        self.sched_model = LRScheduler(
            self.optim_model,
            factor=self.options.train.lr_decay_factor,
            milestones=self.options.train.lr_decay_epochs,
            warmup_steps=self.options.train.lr_warmup_steps,
        )
        if self.options.train.finetune_gan:
            self.optim_discriminator = optim.Adam(
                self.discriminator.parameters(), lr=self.options.train.lr_discriminator
            )
            self.sched_discriminator = LRScheduler(
                self.optim_discriminator,
                factor=self.options.train.lr_decay_factor,
                milestones=self.options.train.lr_decay_epochs,
                warmup_steps=self.options.train.lr_warmup_steps,
            )

        self.sample_fixed_latent = self.model(
            self.visualize_batch_size_per_node * self.world_size,
            device=self.device,
            mode="sample-latent",
        )

        if self.options.model.type == "ecdp":
            # TODO: change this?
            self.model.model.set_generate_steps(300)

        if self.options.train.distributed:
            self.model = DDP(self.model, device_ids=[self.rank])
            dist.broadcast(self.sample_fixed_latent, src=0)

        self.lazy_values = LazyValues(
            epoch_length=lambda: len(self.train_dataloader),
            writer=self._get_writer,
            train_dataloader=lambda: get_train_dataloader(self.options),
            val_dataloader=lambda: get_val_dataloader(self.options),
            test_dataset=lambda: get_test_dataset(self.options),
        )

        if (
            hasattr(self.options.train, "param_running_avg")
            and self.options.train.param_running_avg is not None
        ):
            self.model_avg = make_model(self.options)
            self.model_avg = self.model_avg.to(self.device)
            self.model_avg.load_state_dict(self.model.state_dict())
            self.model_avg.set_image_size(
                (
                    self.options.train.train_image_size,
                    self.options.train.train_image_size,
                )
            )
            self.model_avg.model.set_generate_steps(300)

        self.checkpoint_path = (
            self.options.train.result_dir / "checkpoints" / "checkpoint.pt"
        )
        if self.checkpoint_path.is_file():
            checkpoint = torch.load(
                self.checkpoint_path, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint["model"])
            self.optim_model.load_state_dict(checkpoint["optim_model"])
            self.sched_model.load_state_dict(checkpoint["sched_model"])
            if self.options.train.finetune_gan:
                self.discriminator.load_state_dict(checkpoint["discriminator"])
                self.optim_discriminator.load_state_dict(
                    checkpoint["optim_discriminator"]
                )
                self.sched_discriminator.load_state_dict(
                    checkpoint["sched_discriminator"]
                )
            if (
                hasattr(self.options.train, "param_running_avg")
                and self.options.train.param_running_avg is not None
            ):
                self.model_avg.load_state_dict(checkpoint["model_avg"])
            self.sample_fixed_latent.copy_(checkpoint["sample_fixed_latent"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.lazy_values.set_value("epoch_length", checkpoint["epoch_length"])
        else:
            self.start_epoch = 0

        if self.rank == 0:
            print(
                "Number of parameters:",
                get_number_of_parameters(self.model),
                file=sys.stderr,
            )
            if options.train.distributed:
                print(f"Training with {self.world_size} GPUs", file=sys.stderr)

    def train(self):
        for epoch in range(self.start_epoch, self.options.train.epochs):
            self.train_one(epoch)
            self.evaluate_one(epoch)
            self.sched_model.step_epoch()
            if self.options.train.finetune_gan:
                self.sched_discriminator.step_epoch()
            self.save_state(epoch)
            # if (
            #     self.options.train.test_interval is not None
            #     and (epoch + 1) % self.options.train.test_interval == 0
            # ):
            #     self.test_one(epoch)

    def train_one(self, epoch):
        self.model.train()
        if self.options.train.distributed:
            self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            self.writer.add_scalar(
                "num params",
                get_number_of_parameters(self.model),
                self.global_step(epoch, 0),
            )

        if self.options.train.finetune_percep:
            percep_net = VGGPercepLoss().to(self.device)

        for i, sample in enumerate(tqdm(self.train_dataloader, desc=f"epoch {epoch}")):
            if (
                hasattr(self.options.train, "train_epoch_clip")
                and i >= self.options.train.train_epoch_clip
            ):
                break

            assert not self.options.train.finetune_gan

            if self.options.train.finetune_gan:
                to_train_model = epoch >= 1
                to_train_discriminator = True
            else:
                to_train_model = True
                to_train_discriminator = False

            gen_image = None

            sample = sample_to_device(sample, self.device)
            image = sample["image"]
            image_lr = sample["image_lr"]

            statistics = {}

            if to_train_model:
                self.optim_model.zero_grad(set_to_none=True)

                if not self.options.train.finetune_percep:
                    loss = self.model(image, image_lr, mode="loss")
                    loss = loss.mean()
                    statistics["train loss"] = loss.item()
                    loss.backward()

                if self.options.train.finetune_percep:
                    self.model.model.diffusion.use_ode = True
                    img_gen = self.model(image_lr, t=1.0, mode="random-generate")
                    percep_loss = percep_net(img_gen, image)
                    percep_loss = percep_loss.mean()
                    statistics["train percep loss"] = percep_loss.item()
                    (percep_loss * self.options.train.finetune_percep_weight).backward()
                    self.model.model.diffusion.use_ode = False

                if self.options.train.finetune_pixel:
                    self.model.model.diffusion.use_ode = True
                    img_gen = self.model(image_lr, t=0.0, mode="random-generate")
                    pixel_loss = (img_gen - image).abs().mean(dim=[1, 2, 3])
                    pixel_loss = pixel_loss.mean()
                    statistics["train pixel loss"] = pixel_loss.item()
                    (pixel_loss * self.options.train.finetune_pixel_weight).backward()
                    self.model.model.diffusion.use_ode = False

                if self.options.train.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.options.train.clip_grad_value
                    )

                self.optim_model.step()
                self.sched_model.step()

                self.optim_model.zero_grad(set_to_none=True)

            if to_train_discriminator:
                self.optim_discriminator.zero_grad(set_to_none=True)

                if gen_image is None:
                    with torch.no_grad():
                        gen_image = self.model(image_lr, t=0.9, mode="random-generate")
                real_gan_feat = self.discriminator(image)
                fake_gan_feat = self.discriminator(gen_image)
                real_loss = gan_loss(real_gan_feat, 1.0)
                real_loss = real_loss.mean()
                fake_loss = gan_loss(fake_gan_feat, 0.0)
                fake_loss = fake_loss.mean()
                statistics["train discriminator real loss"] = real_loss.item()
                statistics["train discriminator fake loss"] = fake_loss.item()
                (real_loss + fake_loss).backward()

                self.optim_discriminator.step()
                self.sched_discriminator.step()

                self.optim_discriminator.zero_grad(set_to_none=True)

            self.log_statistics(self.global_step(epoch, i), statistics)

            if (
                hasattr(self.options.train, "param_running_avg")
                and self.options.train.param_running_avg is not None
            ):
                with torch.no_grad():
                    eps = 1 / self.options.train.param_running_avg
                    for param_src, param_dst in zip(
                        self.model.parameters(), self.model_avg.parameters()
                    ):
                        param_dst.mul_(1 - eps)
                        param_dst.add_(param_src, alpha=eps)

    @torch.no_grad()
    def evaluate_one(self, epoch):
        if (
            hasattr(self.options.train, "param_running_avg")
            and self.options.train.param_running_avg is not None
        ):
            my_model = self.model_avg
        else:
            my_model = self.model
        my_model.eval()

        total_statistics = {}
        total_count = 0

        lpips = LPIPS().to(self.device)

        for i, sample in enumerate(
            tqdm(self.val_dataloader, desc=f"epoch {epoch} val")
        ):
            if (
                hasattr(self.options.train, "val_epoch_clip")
                and i >= self.options.train.val_epoch_clip
            ):
                break

            sample = sample_to_device(sample, self.device)
            image = sample["image"]
            image_lr = sample["image_lr"]

            statistics = {}

            loss = my_model(image, image_lr, mode="loss")
            loss = loss.mean()
            statistics["val loss"] = loss.item()

            if epoch % self.options.train.val_images_interval == 0:
                gen = my_model(image_lr, t=1.0, mode="random-generate").clamp(0, 1)
                statistics["psnr diff"] = psnr(image, gen).mean().item()
                statistics["ssim diff"] = ssim(image, gen).mean().item()
                statistics["lpips diff"] = lpips(image, gen).mean().item()
                # print(statistics['psnr diff'], statistics['ssim diff'], statistics['lpips diff'])

            for k, v in statistics.items():
                total_statistics.setdefault(k, 0)
                total_statistics[k] += v * image.shape[0]
            total_count += image.shape[0]

            if i == 0 and epoch % self.options.train.val_images_interval == 0:
                recons = my_model(
                    torch.zeros_like(
                        self.sample_fixed_latent[self.rank :: self.world_size]
                    ),
                    image_lr[: self.visualize_batch_size_per_node],
                    mode="generate",
                ).clamp(0, 1)
                self.log_images(self.global_step(epoch, 0), "generated t=0", recons)

                recons = my_model(
                    self.sample_fixed_latent[self.rank :: self.world_size] * 0.9,
                    image_lr[: self.visualize_batch_size_per_node],
                    mode="generate",
                ).clamp(0, 1)
                self.log_images(self.global_step(epoch, 0), "generated t=0.9", recons)

                recons = my_model(
                    self.sample_fixed_latent[self.rank :: self.world_size],
                    image_lr[: self.visualize_batch_size_per_node],
                    mode="generate",
                ).clamp(0, 1)
                self.log_images(self.global_step(epoch, 0), "generated t=1", recons)

        # print(total_statistics['psnr diff'], total_statistics['ssim diff'], total_statistics['lpips diff'])
        self.log_statistics(
            self.global_step(epoch, 0), total_statistics, weight=total_count
        )

    @torch.no_grad()
    def test_one(self, epoch=None, save_images=False):
        if epoch is None:
            assert self.start_epoch > 0
            epoch = self.start_epoch - 1

        if (
            hasattr(self.options.train, "param_running_avg")
            and self.options.train.param_running_avg is not None
        ):
            my_model = self.model_avg
        else:
            my_model = self.model
        my_model.eval()

        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_count = 0

        lpips = LPIPS().to(self.device)

        inner_model = my_model
        if self.options.train.distributed:
            inner_model = inner_model.module
        inner_model.model.diffusion.use_ode = True

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.options.train.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

        idx_base = 0
        for sample in tqdm(
            loader,
            desc=f"epoch {epoch} test",
        ):
            sample = sample_to_device(sample, self.device)
            image = sample["image"]
            image_lr = sample["image_lr"]

            with inner_model.with_image_size(
                (
                    image_lr.shape[2] * self.options.model.sr_factor,
                    image_lr.shape[3] * self.options.model.sr_factor,
                )
            ):
                gen_diff = inner_model(image_lr, t=1.0, mode="random-generate")
                gen_diff = gen_diff[:, :, : image.shape[2], : image.shape[3]]

            image = quantize(image)
            gen_diff = quantize(gen_diff)
            psnr_diff = psnr(gen_diff, image)
            ssim_diff = ssim(gen_diff, image)
            lpips_diff = lpips(gen_diff, image)

            total_psnr += psnr_diff.sum().item()
            total_ssim += ssim_diff.sum().item()
            total_lpips += lpips_diff.sum().item()
            total_count += image.shape[0]

            if save_images:
                for idx in range(idx_base, idx_base + image.shape[0]):
                    save_path = (
                        self.options.train.result_dir
                        / "images"
                        / f"epoch{epoch}"
                        / f"{idx:02d}.png"
                    )
                    save_image(gen_diff[idx - idx_base], save_path)
                idx_base += image.shape[0]

        statistics = {
            "test psnr": total_psnr,
            "test ssim": total_ssim,
            "test lpips": total_lpips,
        }
        self.log_statistics(
            self.global_step(epoch, 0),
            statistics,
            weight=total_count,
            print_to_stderr=True,
        )

    def save_state(self, epoch):
        if self.rank == 0:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model": self.model.state_dict(),
                "optim_model": self.optim_model.state_dict(),
                "sched_model": self.sched_model.state_dict(),
                "sample_fixed_latent": self.sample_fixed_latent,
                "epoch": epoch,
                "epoch_length": self.epoch_length,
            }
            if self.options.train.finetune_gan:
                checkpoint.update(
                    {
                        "discriminator": self.discriminator.state_dict(),
                        "optim_discriminator": self.optim_discriminator.state_dict(),
                        "sched_discriminator": self.sched_discriminator.state_dict(),
                    }
                )
            if (
                hasattr(self.options.train, "param_running_avg")
                and self.options.train.param_running_avg is not None
            ):
                checkpoint.update(
                    {
                        "model_avg": self.model_avg.state_dict(),
                    }
                )
            torch.save(checkpoint, self.checkpoint_path)
            if epoch % self.options.train.checkpoint_backup_interval == 0:
                torch.save(
                    checkpoint,
                    self.options.train.result_dir
                    / "checkpoints"
                    / f"checkpoint-{epoch}.pt",
                )

    @property
    def epoch_length(self):
        return self.lazy_values.epoch_length

    def global_step(self, epoch, i):
        return epoch * self.epoch_length + i

    @property
    def writer(self):
        return self.lazy_values.writer

    def _get_writer(self):
        if self.options.train.tensorboard and self.rank == 0:
            return SummaryWriter(
                self.options.train.result_dir / "tensorboard",
                purge_step=self.global_step(self.start_epoch, 0),
            )
        else:
            return DummyWriter()

    def flush_writer(self):
        writer = self.lazy_values.get_or_none("writer")
        if writer is not None:
            writer.flush()

    def log_statistics(self, step, statistics, *, weight=1, print_to_stderr=False):
        if self.options.train.distributed:
            keys_sorted = sorted(statistics.keys())
            stats_tensor = torch.tensor(
                [weight] + [statistics[k] for k in keys_sorted],
                dtype=torch.float,
                device=self.device,
            )
            dist.reduce(stats_tensor, dst=0, op=dist.ReduceOp.SUM)
            if self.rank == 0:
                weight = stats_tensor[0].item()
                statistics = {
                    k: stats_tensor[i + 1].item() for i, k in enumerate(keys_sorted)
                }

        if self.rank == 0:
            for k, v in statistics.items():
                self.writer.add_scalar(k, v / weight, step)
            if print_to_stderr:
                print("Testing results:", file=sys.stderr)
                print(
                    yaml.safe_dump({k: v / weight for k, v in statistics.items()}),
                    end="",
                    file=sys.stderr,
                )

    def log_images(self, step, name, images, columns=4):
        if self.options.train.distributed:
            images_dst = [torch.zeros_like(images) for _ in range(self.world_size)]
            dist.all_gather(images_dst, images)
            if self.rank == 0:
                images = torch.stack(images_dst, dim=1).flatten(0, 1)

        if self.rank == 0:
            grid_image = make_grid(images, nrow=columns, padding=0)
            self.writer.add_image(name, grid_image, step)

    @property
    def train_dataloader(self):
        return self.lazy_values.train_dataloader

    @property
    def val_dataloader(self):
        return self.lazy_values.val_dataloader

    @property
    def test_dataset(self):
        return self.lazy_values.test_dataset
