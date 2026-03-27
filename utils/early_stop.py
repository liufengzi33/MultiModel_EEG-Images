class SchedulerEarlyStopper:
    def __init__(self, max_plateaus: int = 4):
        self.max_plateaus = max_plateaus
        self.plateau_count = 0
        self.prev_lr = None

    def step(self, optimizer):
        """Call this after scheduler.step() to update and check if early stop is needed."""
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 第一次调用时记录初始学习率
        if self.prev_lr is None:
            self.prev_lr = current_lr
            return False

        # 如果学习率变小了，记录一次衰减
        if current_lr < self.prev_lr:
            self.plateau_count += 1
            self.prev_lr = current_lr

        # 判断是否达到最大衰减次数
        return self.plateau_count >= self.max_plateaus


class StrictLossEarlyStopper:
    def __init__(self, patience: int = 30, min_delta: float = 0.0, min_epochs: int = 30):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs  # 新增：最小训练轮数
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.current_epoch = 0  # 新增：记录当前轮数

    def step(self, val_loss: float):
        self.current_epoch += 1

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            # 只有当达到设定的耐心值，并且当前轮数大于最小训练轮数时，才触发早停
            if self.counter >= self.patience and self.current_epoch >= self.min_epochs:
                self.early_stop = True

        return self.early_stop