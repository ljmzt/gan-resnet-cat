# references
# https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
# https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
# https://github.com/pfnet-research/sngan_projection/blob/master/updater.py
# https://github.com/EmilienDupont/wgan-gp/blob/master/training.py


import torch
from torchvision.utils import make_grid, save_image
import os
from datetime import datetime
from tqdm import tqdm
import pickle
from torch import nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import logging
  
def getlogger(logfile):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler(logfile, 'w')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class Trainer():
    def __init__(self, config, img_size, Generator, Discriminator, device):
        self.config = config
        self.device = device
        
        if self.config['GAN_type'] in ['GAN', 'WGAN']:
            self.G = Generator(img_size, config['dim'],config['zdim'], use_bn=True)
            self.D = Discriminator(img_size, config['dim'], use_bn=True, use_sn=False)

        elif self.config['GAN_type'] == 'WGAN-GP':
            self.G = Generator(img_size, config['dim'],config['zdim'], use_bn=True)
            self.D = Discriminator(img_size, config['dim'], use_bn=False, use_sn=False)

        elif self.config['GAN_type'] in ['SNGAN', 'SNGAN-hinge']:
            self.G = Generator(img_size, config['dim'],config['zdim'], use_bn=True)
            # caption of fig 8 of Miyato says removed BN which makes sense
            # if applies BN, the location x_r and x_f will be modified, then the gradient won't make sense
            self.D = Discriminator(img_size, config['dim'], use_bn=False, use_sn=True) 

        else:
            print('strange',self.config['GAN_type'])

        self.G = self.G.to(device)
        self.D = self.D.to(device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config['lr_G'], betas=[0.5,0.999])
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config['lr_D'], betas=[0.5,0.999])
#        self.opt_G = torch.optim.SGD(self.G.parameters(), lr=self.config['lr_G'])
#        self.opt_D = torch.optim.SGD(self.D.parameters(), lr=self.config['lr_D'])

        self.log_dir = config['log_dir']
        self.ckpt_dir = config['ckpt_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.z_samples = torch.randn(config['z_samples'],config['zdim'],device=device)
    
    def resume(self, savefile):
        ckpt = torch.load(savefile)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.opt_G.load_state_dict(ckpt['opt_G'])
        self.opt_D.load_state_dict(ckpt['opt_D'])
    
    def inference(self, z):
        self.G.eval()
        with torch.no_grad():
            f = self.G(z)
        return f
    
    # needs to modify this bit for types of data
    def evaluate(self, log_dir, epoch):
        self.G.eval()
        with torch.no_grad():
            imgs = self.G(self.z_samples)
        grid_img = make_grid(imgs.cpu(), nrow=10)
        save_image(grid_img, os.path.join(log_dir,f"epoch_{epoch}.jpg"))
        plt.figure(figsize=(10,10))
        plt.imshow(grid_img.permute(1,2,0))
        plt.show()
    
    def train(self, dataset):
        # log this training
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.join(self.log_dir, f"_{time}")
        os.makedirs(log_dir)
        logger = getlogger(os.path.join(log_dir, 'log.txt'))
        with open(os.path.join(log_dir,"config.pkl"),'wb') as fid:
            pickle.dump(self.config, fid)
        
        # check point
        ckpt_dir = os.path.join(self.ckpt_dir, f"_{time}")
        os.makedirs(ckpt_dir, exist_ok=True)  # can be the same as log_dir
        
        # training stuff
        loss = nn.BCEWithLogitsLoss()
        if (self.device == 'cuda'):
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        
        for epoch in range(self.config['epoches']):
            progress_bar = tqdm(dataloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            progress_bar.set_description(f"epoch {epoch}")
            self.G.train()
            self.D.train()
            
            loss_D_sum, n_D, loss_G_sum, n_G = 0.0, 0, 0.0, 0
            grad_norm_sum = 0.0
            
            for i, r in enumerate(progress_bar):
            
                r = r.to(self.device)
                bs = r.shape[0]

                f_label = torch.zeros(bs, device=self.device)
                r_label = torch.ones(bs, device=self.device)
                z = torch.randn(bs, self.config['zdim'], device=self.device)
             
                ### training D ###
                if (i % self.config['D_step'] == 0):                
                    for p in self.D.parameters():
                        p.requires_grad = True

                    f = self.G(z)                    
                    if self.config['GAN_type'] in ['GAN']:
                        loss_r = loss(self.D(r), r_label)
                        loss_f = loss(self.D(f), f_label)
                        loss_D = 0.5 * (loss_r + loss_f)
                        loss_D_sum += loss_D.item() * bs
                        eps = torch.rand(bs,1,1,1,device=self.device).expand_as(r)
                        interpolation = eps * r + (1.0 - eps) * f
                        logit = self.D(interpolation)
                        grad_outputs = torch.ones_like(logit, device=self.device)
                        gradients = torch.autograd.grad(outputs = logit,
                                            inputs = interpolation,
                                            grad_outputs = grad_outputs,
                                            create_graph = True,
                                            retain_graph = True)[0]
                        grad_norm = gradients.reshape(bs,-1).norm(2,dim=1)
                        grad_norm = torch.mean((grad_norm-1)**2)
                        grad_norm_sum += grad_norm.item() * bs
                    
                    elif self.config['GAN_type'] == 'WGAN':
                        loss_r = torch.mean(self.D(r), dim=-1)
                        loss_f = torch.mean(self.D(f), dim=-1)
                        loss_D = -loss_r + loss_f
                        loss_D_sum += loss_D.item() * bs
                        
                    elif self.config['GAN_type'] == 'WGAN-GP':
                        loss_r = torch.mean(self.D(r), dim=-1)
                        loss_f = torch.mean(self.D(f), dim=-1)
                        eps = torch.rand(bs,1,1,1,device=self.device).expand_as(r)
                        interpolation = eps * r + (1.0 - eps) * f
                        logit = self.D(interpolation)
                        grad_outputs = torch.ones_like(logit, device=self.device)
                        gradients = torch.autograd.grad(outputs = logit,
                                            inputs = interpolation,
                                            grad_outputs = grad_outputs,
                                            create_graph = True,
                                            retain_graph = True)[0]
                        grad_norm = gradients.reshape(bs,-1).norm(2,dim=1)
                        grad_norm = torch.mean((grad_norm-1)**2)
                        grad_norm_sum += grad_norm.item() * bs
                        loss_D1 = -loss_r + loss_f
                        loss_D_sum += loss_D1.item() * bs
                        loss_D = loss_D1 + self.config['lambda']*grad_norm

                    elif self.config['GAN_type'] == 'SNGAN':
                        loss_r = loss(self.D(r), r_label)
                        loss_f = loss(self.D(f), f_label)
                        loss_D = 0.5 * (loss_r + loss_f)
                        #loss_r = torch.mean(self.D(r), dim=-1)
                        #loss_f = torch.mean(self.D(f), dim=-1)
                        #loss_D = -loss_r + loss_f
                        loss_D_sum += loss_D.item() * bs
                        eps = torch.rand(bs,1,1,1,device=self.device).expand_as(r)
                        interpolation = eps * r + (1.0 - eps) * f
                        logit = self.D(interpolation)
                        grad_outputs = torch.ones_like(logit, device=self.device)
                        gradients = torch.autograd.grad(outputs = logit,
                                            inputs = interpolation,
                                            grad_outputs = grad_outputs,
                                            create_graph = True,
                                            retain_graph = True)[0]
                        grad_norm = gradients.reshape(bs,-1).norm(2,dim=1)
                        grad_norm = torch.mean((grad_norm-1)**2)
                        grad_norm_sum += grad_norm.item() * bs

                    elif self.config['GAN_type'] == 'SNGAN-hinge':
                        loss_r = torch.mean(nn.ReLU()(1.0-self.D(r)), dim=-1)
                        loss_f = torch.mean(nn.ReLU()(1.0+self.D(f)), dim=-1)
                        loss_D = loss_r + loss_f
                        loss_D_sum += loss_D.item() * bs
#                         eps = torch.rand(bs,1,1,1,device=self.device).expand_as(r)
#                         interpolation = eps * r + (1.0 - eps) * f
#                         logit = self.D(interpolation)
#                         grad_outputs = torch.ones_like(logit, device=self.device)
#                         gradients = torch.autograd.grad(outputs = logit,
#                                             inputs = interpolation,
#                                             grad_outputs = grad_outputs,
#                                             create_graph = True,
#                                             retain_graph = True)[0]
#                         grad_norm = gradients.reshape(bs,-1).norm(2,dim=1)
#                         grad_norm = torch.mean((grad_norm-1)**2)
#                         grad_norm_sum += grad_norm.item() * bs


                    self.opt_D.zero_grad()
                    loss_D.backward()
                    self.opt_D.step()
                    n_D += bs
                    
                    if self.config['GAN_type'] == 'WGAN':
                        for p in self.D.parameters():
                            p.data.clamp_(-self.config['clamp_value'], self.config['clamp_value'])
                
                
                ### training G ###
                if (i % self.config['G_step'] == 0):
                    for p in self.D.parameters():
                        p.requires_grad = False

                    f = self.G(z)
                    if self.config['GAN_type'] in ['GAN', 'SNGAN']:
                        loss_G = loss(self.D(f), r_label)
                    elif self.config['GAN_type'] in ['WGAN', 'WGAN-GP', 'SNGAN-hinge']:
                        loss_G = -torch.mean(self.D(f), dim=-1)

                    if (self.config['kick']) and (epoch <= 10):
                        f_std = f.reshape(f.shape[0],-1).std(dim=0).mean()
                        loss_G = loss_G - 10*f_std
                    
                    self.opt_G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                    
                    loss_G_sum += loss_G.item() * bs
                    n_G += bs
                
                if (i % 10 == 0):
                    if self.config['GAN_type'] == 'WGAN-GP':
                        progress_bar.set_postfix({"loss_G":loss_G_sum/n_G, "loss_D":loss_D_sum/n_D, 
                                                  "grad_norm":grad_norm_sum/n_D})
                    else:
                        progress_bar.set_postfix({"loss_G":loss_G_sum/n_G, "loss_D":loss_D_sum/n_D})
                        
            if self.config['GAN_type'] == 'WGAN-GP':
                logger.info(f"epoch{epoch} loss: {loss_G_sum/n_G} {loss_D_sum/n_D} {grad_norm_sum/n_D}")
            else:
                logger.info(f"epoch{epoch} loss: {loss_G_sum/n_G} {loss_D_sum/n_D}")
            self.evaluate(log_dir, epoch)
            
            if (epoch % 10 == 0):
                torch.save({'G':self.G.state_dict(), 'D':self.D.state_dict(),
                            'opt_G': self.opt_G.state_dict(), 'opt_D': self.opt_D.state_dict()}, 
                           os.path.join(ckpt_dir, f'model_{epoch}.pickle'))
