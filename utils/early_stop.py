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
