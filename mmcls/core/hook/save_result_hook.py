
import os.path as osp

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, master_only


@HOOKS.register_module()
class SaveResultHook(Hook):
    """Save the best result to output/result.csv
    """    

    @master_only
    def after_run(self, runner):
        result_file = osp.join(osp.dirname(runner.work_dir), 'result.txt')
        # print('hook msgs', runner.meta['hook_msgs'])
        with open(result_file, 'a') as f:
            f.write(f"{runner.work_dir}\t{runner.meta['hook_msgs']['best_score']}\n")
        
