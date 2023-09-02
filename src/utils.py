import os
import re
import signal
import threading
import psutil
import wandb
import functools


class WandbKiller:
    def __init__(self):
        self.timer = threading.Timer(60, self.force_finish_wandb)

    def force_finish_wandb(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), '../wandb/latest-run/logs/debug-internal.log'), 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Failed to open log file: {e}")
            return

        if not lines:
            print("Log file is empty.")
            return

        last_line = lines[-1]
        match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
        if match:
            pid = int(match.group(2))
            try:
                p = psutil.Process(pid)
                if 'wandb' in ' '.join(p.cmdline()):
                    print(f'wandb pid: {pid}')
                else:
                    print('Process is not a wandb process.')
                    return
            except psutil.NoSuchProcess:
                print('Process does not exist.')
                return
        else:
            print('Cannot find wandb process-id.')
            return

        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Process with PID {pid} killed successfully.")
        except OSError:
            print(f"Failed to kill process with PID {pid}.")

    def try_finish_wandb(self):
        self.timer.start()
        wandb.finish()
        self.timer.cancel()


def to_camel_case(string):
    words = string.split('-')
    camel_case = ''.join([w.capitalize() for w in words])
    return camel_case


def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None
    return wrapper


def get_value_from_locals(var_name, local_vars, transform_func=str, default_value=None):
    value = local_vars.get(var_name, default_value)
    if isinstance(value, (list, tuple)):
        return [transform_func(item) for item in value]
    return value
