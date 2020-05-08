import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import os
import argparse
import yaml
import shutil
import numpy as np

from nets.mlp import Net

from metrics.metrics import Metrics

class Experiment():

    def __init__(self, settings):

        self.logparams = settings['log-params']
        self.hyperparams = settings['hyperparams']
        self.network_config = settings['network-config']
        self.experiment_params = settings['experiment-params']

        self.num_runs = settings['num-runs']
        self.num_epochs = settings['epochs']

        self._generate_output_directory()
        
        # Create data loaders
        use_cuda = torch.cuda.is_available() and not settings['no-cuda']
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            print('GPU accelerated training enabled')
        self.train_loader, self.test_loader = self._create_data_loaders(use_cuda)

        self.network_generator = Net

    def _create_data_loaders(self, use_cuda):
        '''
            Creates the training and test data loaders
            Nothing to see here
        '''
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=self.hyperparams['batch-size'], shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=self.hyperparams['test-batch-size'], shuffle=True, **kwargs)
        
        return train_loader, test_loader
    
    def _generate_output_directory(self):
        '''
            Function to generate the output folder name for the logs
            Obviously, I have tried and failed to keep this sane and manageable
            And then created this function to hide the madness
        '''

        dirname = 'dni' if self.network_config['dni']['flag'] else 'vanila'
        if self.network_config['synth-grad-frozen']['flag'] and self.network_config['dni']['flag']:
            description = 'frozen_synthesizer_nonzero'
            if self.network_config['synth-grad-frozen']['pretrained']:
                description += '_pretrained'
            dirname += '_' + description

        self.logdir = os.path.join(self.logparams['logdir'], dirname)
        os.makedirs(self.logdir, exist_ok=True)

    def _setup_metrics(self):
        self.metrics_list = {}
        if self.logparams['metrics']['loss']:
            self.metrics_list['loss'] = Metrics('loss_curve')
        if self.logparams['metrics']['accuracy']:
            self.metrics_list['accuracy'] = Metrics('acc_curve')
        if self.logparams['metrics']['test-accuracy']:
            self.metrics_list['test_accuracy'] = Metrics('test_accuracy')
        if self.logparams['metrics']['synth-grad-norm']:
            self.metrics_list['grad_norm'] = Metrics('synth_grad_norm')
        if self.logparams['metrics']['weights']:
            for i in range(self.num_runs):
                os.makedirs(os.path.join(self.logdir, 'weight_history', 'run_' + str(i)), exist_ok=True)

    def run_experiment(self):

        for i in range(self.num_runs):

            self.current_run = i
            print('Beginning run ' + str(i))
            self._setup_metrics()

            trained_net_file = self.network_config['synth-grad-frozen']['path'] if self.network_config['synth-grad-frozen']['pretrained'] else None
            self.net = self.network_generator(use_dni=self.network_config['dni']['flag'], context=self.network_config['dni']['context'],
                freeze_synthesizer=self.network_config['synth-grad-frozen']['flag'], device=self.device, trained_net_file=trained_net_file)
            
            print('Model definition')
            print(self.net)

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.hyperparams['learning-rate'])
            # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
            
            self._train(self.num_epochs)
            for metric in self.metrics_list:
                self.metrics_list[metric].save_to_disk(self.logdir, i)

    def _train(self, num_epochs, start_epoch=0, teacher_training=False):
        self.test_acc = []
        self.cosine_dist_hists = []

        log_interval = self.logparams['log-interval']

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            self.net.train()

            # if self.scheduler is not None:
            #     self.scheduler.step()

            running_loss = 0.0
            running_acc = 0.0
            
            if self.logparams['metrics']['weights'] and not teacher_training:
                torch.save(self.net, os.path.join(self.logdir, 'weight_history', 'run_' + str(self.current_run), 'epoch_'+ str(epoch + start_epoch +  1) + '.pt'))

            for batch_idx, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                acc = torch.sum(torch.argmax(outputs, dim=1)==labels).item()/self.train_loader.batch_size
                running_acc += acc

                if batch_idx % log_interval == (log_interval - 1):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining accuracy: {:.2f}%'.format(
                        epoch + start_epoch + 1, batch_idx * self.train_loader.batch_size, len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item(), acc*100))
                        
                    if self.logparams['metrics']['loss']:
                        self.metrics_list['loss'].log_vals(loss.item())
                    if self.logparams['metrics']['accuracy']:
                        self.metrics_list['accuracy'].log_vals(100.0*acc)
                    if self.logparams['metrics']['synth-grad-norm']:
                        if self.network_config['dni']['flag']:
                            self.metrics_list['grad_norm'].log_vals(self.net.backward_interface.synth_grad_norm)

                    running_loss = 0.0
                    running_acc = 0.0

            if self.test_loader is not None:
                val_loss, val_acc = self._test()
                print('End of epoch {}: Validation loss: {:.6f}\tValidation accuracy: {:.2f}%'.format(
                    epoch + start_epoch + 1, val_loss, val_acc*100))
                
                if self.logparams['metrics']['test-accuracy']:
                    self.metrics_list['test_accuracy'].log_vals(val_acc*100)

    def _test(self):
        val_acc = 0.0
        val_loss = 0.0

        self.net.eval()

        for batch_idx, data in enumerate(self.test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # inputs = inputs.view(inputs.shape[0], -1)
            outputs = self.net(inputs)
            
            batch_loss = self.criterion(outputs, labels)

            val_loss += batch_loss.item()
            val_acc += torch.sum(torch.argmax(outputs, dim=1)==labels).item()/self.test_loader.batch_size
        
        val_loss /= len(self.test_loader)
        val_acc /= len(self.test_loader)
        
        return val_loss, val_acc

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DNI experiment settings')
    parser.add_argument('--experiment-file', type=str, default='experiment_settings.yaml',
                        help='settings file for experiment')
    args = parser.parse_args()

    with open(args.experiment_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    torch.manual_seed(settings['random-seed'])
    experiment = Experiment(settings)
    experiment.run_experiment()
    shutil.copy2(args.experiment_file, experiment.logdir)
