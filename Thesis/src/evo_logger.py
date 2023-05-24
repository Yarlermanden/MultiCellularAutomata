from evotorch.logging import PandasLogger, StdOutLogger

class Custom_Logger(PandasLogger):
    def __init__(self, searcher, online_tracker):
        super().__init__(searcher)
        self.online_tracker = online_tracker

    def _log(self, status):
        super()._log(status)
        if 'stepsize' in status:
            status['stepsize'] = status['stepsize'].detach().cpu().item()
        self.online_tracker.log(**status)