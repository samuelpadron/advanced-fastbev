import os
import time
import json
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class DeadlineHook(Hook):
    def __init__(self, deadline_env='TRAIN_DEADLINE', safety_factor=1.05, min_buffer=300):
        self.deadline = float(os.environ.get(deadline_env, 0)) or None
        self.safety_factor = safety_factor
        self.min_buffer = min_buffer
        self.epoch_start = None
        self.timing_file = None
        self.last_duration = None

    def before_run(self, runner):
        self.timing_file = os.path.join(runner.work_dir, '.epoch_times.json')
        if os.path.exists(self.timing_file):
            with open(self.timing_file) as f:
                durations = json.load(f)
            if durations:
                self.last_duration = max(durations[-3:])  # use recent max, not average

    def before_train_epoch(self, runner):
        now = time.time()
        if self.deadline and self.last_duration:
            needed = self.last_duration * self.safety_factor + self.min_buffer
            remaining = self.deadline - now
            if remaining < needed:
                runner.logger.info(
                    f'Only {remaining/3600:.2f}h left, need ~{needed/3600:.2f}h '
                    'for another epoch. Exiting cleanly for resubmission.')
                open(os.path.join(runner.work_dir, '.need_resubmit'), 'w').close()
                raise SystemExit(0)
        self.epoch_start = now

    def after_train_epoch(self, runner):
        if self.epoch_start is None:
            return
        duration = time.time() - self.epoch_start
        durations = []
        if os.path.exists(self.timing_file):
            with open(self.timing_file) as f:
                durations = json.load(f)
        durations.append(duration)
        with open(self.timing_file, 'w') as f:
            json.dump(durations, f)