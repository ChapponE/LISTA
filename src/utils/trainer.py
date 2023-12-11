import time
import torch

class Trainer():
    def __init__(self,
                 model, optimizer, scheduler, loss,
                 train_dataloader, val_dataloader=None, check_val_every_n_epoch=1, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.metrics = Metrics()
        self.history = {}
        self.epoch = 0
        self.verbose = verbose

    def run(self, num_epoch):
        """ Run training until the given number of epochs is reached """
        try:
            # Progress bar
            from tqdm.notebook import tqdm
            training_pbar = tqdm(
                iterable=range(self.epoch, num_epoch),
                initial=self.epoch,
                total=num_epoch,
                desc="Training",
                unit="epoch"
            )

            for _ in training_pbar:
                self.training_epoch()
                ### test ###
                # print('l 35 trainer', list(self.model.parameters())[2])
                ### test ###
                training_pbar.set_postfix(self.history["Training"][-1][1])
                if self.epoch % self.check_val_every_n_epoch == 0:
                    self.validation_epoch()
                    epoch, metrics = self.history["Validation"][-1]
                    if self.verbose:
                        print(f"Validation epoch {epoch:4d} | " + " | ".join(
                            (f"{k}: {v:.2e}" for k, v in metrics.items())))


        except KeyboardInterrupt:
            """ Stop training if Ctrl-C is pressed """
            pass

    def training_epoch(self):
        """ Train for one epoch """
        device = self.device
        self.metrics.init()
        for x, target, *_ in self.train_dataloader:
            x = x.to(device)
            target = target.to(device)

            y = self.model(x)
            loss = self.loss(y, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.threshold.data = self.model.threshold.data.clamp(min=0)
            self.metrics.accumulate(loss, x, y, target)

        self.epoch += 1
        self.log("Training", self.epoch, self.metrics.summarize())

    def validation_epoch(self):
        """ Run a validation step """
        if self.val_dataloader is None:
            return

        device = self.device
        self.metrics.init()
        with torch.no_grad():
            for x, target, *_ in self.val_dataloader:
                x = x.to(device)
                target = target.to(device)

                y = self.model(x)
                loss = self.loss(y, target)
                if self.scheduler != False:
                    self.scheduler.step(loss)  # ajout
                self.metrics.accumulate(loss, x, y, target)

        self.log("Validation", self.epoch, self.metrics.summarize())

    def log(self, mode, epoch, metrics):
        history = self.history.setdefault(mode, [])
        history.append((epoch, metrics))
        # print(f"{mode} epoch {epoch:4d} | " + " | ".join((f"{k}: {v:.2e}" for k, v in metrics.items())))

    @property
    def device(self):
        """ Training device defined from the device of the first parameter of the model """
        return next(self.model.parameters()).device


class Metrics:
    """ Calculate metrics from training and validation steps """

    def init(self):
        self.metrics = dict()
        self.cnt = 0
        self.tic = time.perf_counter()

    def accumulate(self, loss, x, y, target):
        with torch.no_grad():
            self.metrics["Loss"] = self.metrics.get("Loss", 0) + loss.item()
            self.metrics["PSNR"] = self.metrics.get("PSNR", 0) + PSNR(y, target).mean().item()
            # self.metrics["SNR"] = self.metrics.get("SNR", 0) + SNR(y, target).mean().item()
        self.cnt += 1

    def summarize(self):
        self.toc = time.perf_counter()
        metrics = {k: v / self.cnt for k, v in self.metrics.items()}
        metrics["Wall time"] = self.toc - self.tic
        return metrics

def PSNR(img1, img2):
    # Optional: get last 3 dimensions (CxNxM) so that to be compatible with single sample
    #dims = list(range(max(0, img1.ndim - 3), img1.ndim))
    # otherwise simply
    dims = list(range(1, img1.ndim))

    # Take care to the mean along only the last 3 dimensions.
    return 10 * torch.log10(1. / (img1 - img2).pow(2).mean(dims))

def SNR(img1, img2):
    dims = list(range(1, img1.ndim))
    return 20 * torch.log2(1. / (img1 - img2).pow(2).mean(dims))

