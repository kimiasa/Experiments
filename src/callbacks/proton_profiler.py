import triton.profiler as proton
from pytorch_lightning import Callback, Trainer, LightningModule

class ProtonProfiler(Callback):
    def __init__(self, profile_name="model_profile", context="python", profile_first_epoch_only=True):
        super().__init__()
        self.profile_name = profile_name
        self.context = context
        self.profile_first_epoch_only = profile_first_epoch_only
        self.session_id = None
        self.has_profiled = False  # Flag to check if profiling has been done

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        # Start the Proton profiler session when training begins
        self.session_id = proton.start(name=self.profile_name, context=self.context)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx):
        if not self.has_profiled:
            proton.activate(self.session_id)
        with proton.scope("forward_pass"):
            pass  

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule, batch_idx):
        if not self.has_profiled:
            with proton.scope("backward_pass"):
                pass  
            proton.deactivate(self.session_id)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs):
        if self.profile_first_epoch_only and not self.has_profiled:
            if trainer.current_epoch == 0: 
                proton.finalize(self.session_id)
                self.has_profiled = True 

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.session_id and not self.has_profiled:
            proton.finalize(self.session_id)
