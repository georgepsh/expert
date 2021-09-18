from datalib.dataflow import get_dataflows
from datalib.torch_dataset import DemoDataset, DemoLoader


class ExpertRunner(object):
    def __init__(self, env, model, config):
        self.env = env
        self.model = model
        # self.dataflow_iter = get_dataflows(config).get_data()
        dataset = DemoDataset(config['demos_path'])
        self.dataflow_iter = DemoLoader(dataset, config['batch_size'])

    def run(self):
        """
        Return a batch of
        - observations
        - values according to the freshest value function estimate
        - expert actions
        """
        obs, actions, future_discounted_rewards = next(self.dataflow_iter)
        values = self.model.expert_train_model.value(obs)
        return obs, actions, future_discounted_rewards, values
