# Torch Training Job

*This package is mainly for personal use. It is extracted from one of my private projects so that it can be maintained independently.*

*It almost certainly does not work well for now.*

This is a package that provides an OO style Training Job.

It aims to provide a common interface for different ML training job.

To use this package, you need to define a job class of your own which inherits TrainingJob. 
Usually you would need a config file
(only `yaml` is supported for now).

```python
from torch_training_job import TrainingJob

class MyTrainingJob(TrainingJob):

    def get_model(self, model_config):
        return mymodel(config)

    def load_data(self, data_config):
        train_loader = ...
        test_loader = ...
        return train_loader, test_loader
    
    def run(self, *args, **kwargs):
        """
        Main logic of your job
        """
        ...
    
    ...
```

Your entry point should be like this:

```python
job = MyTrainingJob(config)
job.run()
```

## TODO

- Make it a typical python package that can be installed by pip etc.
- Write documentation.
- Add a cookbook.
- improve default `run` method