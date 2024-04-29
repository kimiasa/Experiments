import triton.profiler as proton
from pytorch_lightning import Callback, Trainer, LightningModule

from typing import Any

class ProtonProfiler(Callback):
    def __init__(self, profile_name="model_profile", context="shadow", profile_first_batch_only=True):
        super().__init__()
        self.profile_name = profile_name
        self.context = context
        self.profile_first_batch_only = profile_first_batch_only
        self.session_id = None
        self.has_profiled = False  

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        self.session_id = proton.start(name=self.profile_name, context=self.context)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx):
        if not self.has_profiled:
            proton.activate(self.session_id) 

    #def on_after_backward(self, trainer: Trainer, pl_module: LightningModule, batch_idx):
    #    if not self.has_profiled:
    #        proton.deactivate(self.session_id)

    #def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs):
    #    if self.profile_first_epoch_only and not self.has_profiled:
    #        if trainer.current_epoch == 0: 
    #            proton.finalize(self.session_id)
    #            self.has_profiled = True 

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *_: Any) -> None:
        if self.profile_first_batch_only and not self.has_profiled:
            proton.finalize(self.session_id)
            self.has_profiled = True  


    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.session_id and not self.has_profiled:
            proton.finalize(self.session_id)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        self.test_session_id = proton.start(name=f"{self.profile_name}_test", context=self.context)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.test_session_id:
            proton.finalize(self.test_session_id)
            self.test_session_id = None
