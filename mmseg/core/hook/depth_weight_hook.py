from mmcv.runner.hooks.hook import Hook, HOOKS

@HOOKS.register_module()
class DepthWeightHook(Hook):
    def __init__(self, depth_weight_start=0):
        self.depth_weight_start = depth_weight_start

    def set_depth_weight(self, runner):
        depth_reweight = False
        if self.depth_weight_start < 0:
            depth_reweight = False
        elif runner.iter >= self.depth_weight_start:
            depth_reweight = True
        else:
            depth_reweight = False
        
        runner.model.depth_reweight = depth_reweight
        # runner.logger.info(f'model depth_reweight changed to {depth_reweight}')

    def before_train_iter(self, runner):
        self.set_depth_weight(runner)

    def before_tr(self, runner):
        self.set_depth_weight(runner)

    def before_run(self, runner):
        self.set_depth_weight(runner)
