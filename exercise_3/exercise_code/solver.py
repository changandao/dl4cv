from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################      
        
        num_iterations = num_epochs * iter_per_epoch
        inputs = []
        targets = []
        for i, data in enumerate(train_loader,0):
            input, target = Variable(data[0]), Variable(data[1])
            if model.is_cuda:
                input,target = input.cuda(), target.cuda()
            inputs.append(input)
            targets.append(target)
        t = 0
        for epoch in range(num_epochs):
            # epoch_end = (t + 1) % iter_per_epoch == 0
            # if epoch_end:
            #     epoch += 1
            #
            #     for k in self.optim_configs:
            #         self.optim_configs[k]['learning_rate'] *= self.lr_decay
            train_accs = []
            for i, (input, target) in enumerate(train_loader, 0):
                if model.is_cuda:
                    input, target = input.cuda(), target.cuda()
                #input, target = input, Variable(target)

                optim.zero_grad()
                output = model(input)
                loss = self.loss_func(output, target)
                loss.backward()
                optim.step()
                self.train_loss_history.append(loss.data.cpu().numpy())
                #print(input.data.cpu().numpy().size)
                t = epoch * iter_per_epoch + i
                if t % log_nth == 0:
                    print('(Iteration %d / %d) loss: %f' % (
                        t + 1, num_iterations, self.train_loss_history[-1]))
                if i == iter_per_epoch-1:
                    _, pred = torch.max(output, 1)
                    train_acc = np.mean((pred == target).data.cpu().numpy())
                    self.train_acc_history.append(train_acc)
                    print('[Epoch %d/%d] TRAIN   acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, train_acc, self.train_loss_history[-1]))

            val_losses = []
            val_accs = []
            #model.eval()
            for input, target in val_loader:
                if model.is_cuda:
                    input, target = input.cuda(), target.cuda()
                input, target = input, target

                output = model(input)
                loss = self.loss_func(output, target)
                val_losses.append(loss.data.cpu().numpy())
                _, pred = torch.max(output, 1)
                val_acc = np.mean((pred == target).data.cpu().numpy())

                val_accs.append(val_acc)
                
                

            #model.train()
            val_acc, val_loss = np.mean(val_accs), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, val_acc, val_loss))
         

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
