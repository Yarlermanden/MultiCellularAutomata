import wandb

class OnlineTracker(object):
    def __init__(self, settings):
        self.config = {
                'settings': settings.__dict__,
                'train_config': settings.train_config.__dict__,
            }
        for i in range(len(settings.food_envs)):
            self.config['food_envs' + str(i)] = settings.food_envs[i].__dict__
        wandb.init(
            project="gnca",
            config=self.config
        )
    
    def log(self, **kwargs):
        wandb.log(kwargs)
    
    def add_model(self, model):
        self.config['model'] = model.state_dict()
        wandb.config.update(self.config)

    def finish(self):
        wandb.finish()