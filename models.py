import networks
import os
import math
import torch
import torch.nn as nn
import itertools
import numpy as np


class ZSDAModel():
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_epochs = cfg.n_epochs
        self.epoch = 0
        self.model_names = ['G_D', 'G_T', 'D_D', 'FS', 'C_R', 'C_IR']
        self.device = torch.device('cuda:{}'.format(cfg.gpu_ids[0])) if cfg.gpu_ids else torch.device('cpu')

        if hasattr(cfg, 'Generator') and 'resnet' in cfg.Generator:
            self.netG_D = networks.init_net(networks.resnetFeatureExtractor(pretrained=False, name=cfg.Generator), init_type=None, gpu_ids=cfg.gpu_ids)
            self.netG_T = networks.init_net(networks.resnetFeatureExtractor(pretrained=True, name=cfg.Generator), init_type=None, gpu_ids=cfg.gpu_ids)
            feature_size = self.netG_T.module.feature_size
            nc = feature_size
            resnet = True
        else:
            avgPool = False
            if cfg.rt_data == 'NIST':
                avgPool = True
            self.netG_D = networks.init_net(networks.FeatureExtractor(input_nc=3, output_nc=128, avgPool=avgPool), gpu_ids=cfg.gpu_ids)
            self.netG_T = networks.init_net(networks.FeatureExtractor(input_nc=3, output_nc=128, avgPool=avgPool), gpu_ids=cfg.gpu_ids)
            feature_size = 128 * (round(round(cfg.img_size/2)/2))**2
            nc = 128
            resnet = False

        if cfg.FS == 'FC':
            feature_size = 128 * 9

        self.GRL = networks.init_net(networks.GRL())
        self.netD_D = networks.init_net(networks.Discriminator(input_nc=feature_size, resnet=resnet), gpu_ids=cfg.gpu_ids)
        self.netC_R = networks.init_net(networks.Classifier(input_nc=feature_size, output_nc=cfg.rt_classes, resnet=resnet), gpu_ids=cfg.gpu_ids)
        self.netC_IR = networks.init_net(networks.Classifier(input_nc=feature_size, output_nc=cfg.irt_classes, resnet=resnet), gpu_ids=cfg.gpu_ids)
        self.pool = networks.init_net(nn.AdaptiveAvgPool2d(1), gpu_ids=cfg.gpu_ids)
        
        self.fs_loss_coeff = {'irt_s': 1, 'irt_t': 1, 'rt_s': 2}
        self.cls_loss_coeff = {'irt_s': 0, 'irt_t': 0, 'rt_s': 2}

        if cfg.FS == 'FC':
            self.netFS = networks.init_net(
                networks.FeatureShifter_FC(input_nc=feature_size*2, output_nc=feature_size, hidden_size=feature_size, n_layers=1),
            gpu_ids=cfg.gpu_ids)
        elif cfg.FS == 'SA':
            self.netFS = networks.init_net(networks.FeatureShifter_Att(input_nc=nc), gpu_ids=cfg.gpu_ids)

        self.optG_T = torch.optim.Adam(self.netG_T.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optG_D = torch.optim.Adam(self.netG_D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optD_D = torch.optim.Adam(self.netD_D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optC_R = torch.optim.Adam(self.netC_R.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optC_IR = torch.optim.Adam(self.netC_IR.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        self.optFS = torch.optim.Adam(self.netFS.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

        self.optimizers = self._get_opts(self.model_names)
        self.schedulers = [networks.get_scheduler(optimizer, cfg) for optimizer in self.optimizers]

        self.criterion_xent = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.criterion_adv = nn.BCEWithLogitsLoss().to(self.device)
        
        self.losses = {}
        self.counts = {}

        self.save_dir = os.path.join(cfg.checkpoints_dir, cfg.name)


    def set_input(self, irt_s, irt_t, rt_s):
        self.x = {'irt_s': irt_s[0].to(self.device),
                  'irt_t': irt_t[0].to(self.device), 
                  'rt_s': rt_s[0].to(self.device)}
        self.y = {'irt_s': irt_s[1].to(self.device),
                  'irt_t': irt_t[1].to(self.device), 
                  'rt_s': rt_s[1].to(self.device)}


    def set_pair_input(self, irt_s, irt_t, rt_s):
        self.B = irt_s[0]['anchor'].shape[0]
        irt_s_x = torch.cat((irt_s[0]['anchor'], irt_t[0]['negative']), 0)
        irt_s_y = torch.cat((irt_s[1]['anchor'], irt_t[1]['negative']), 0)
        irt_t_x = torch.cat((irt_s[0]['positive'], irt_t[0]['anchor']), 0)
        irt_t_y = torch.cat((irt_s[1]['positive'], irt_t[1]['anchor']), 0)

        self.x = {'irt_s': irt_s_x.to(self.device),
                  'irt_t': irt_t_x.to(self.device),
                  'rt_s': rt_s[0].to(self.device)}
        self.y = {'irt_s': irt_s_y.to(self.device),
                  'irt_t': irt_t_y.to(self.device),
                  'rt_s': rt_s[1].to(self.device)}


    def _get_opts(self, model_names):
        opts = []
        for name in model_names:
            opt = getattr(self, 'opt' + name)
            opts.append(opt)
            
        return opts


    def update(self):
        # Old version
        # self.class_disentangle()
        # self.update_FS()

        # Paper version
        self.domain_disentangle()
        self.class_disentangle()
        self.collab_learning()


    def domain_disentangle(self):
        updated_models = ['C_R', 'C_IR', 'D_D', 'G_T', 'G_D']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss = 0
        for task in self.x:
            f_ci = self.netG_D(self.x[task])
            f_di = self.netG_T(self.x[task])
            
            
            task_loss = 0
            if task == 'rt_s':
                di_class_pred = self.netC_R(f_di)
            else:
                di_class_pred = self.netC_IR(f_di)
            task_loss += self.criterion_xent(di_class_pred, self.y[task]) * self.fs_loss_coeff[task] / 3

            domain_pred = self.netD_D(self.GRL(f_di))
            ci_domain_pred = self.netD_D(f_ci)
            if task == 'irt_t':
                domain_loss = self.criterion_adv(ci_domain_pred, torch.ones(ci_domain_pred.shape).to(self.device)) + \
                              self.criterion_adv(domain_pred, torch.ones(domain_pred.shape).to(self.device))
            else:
                domain_loss = self.criterion_adv(ci_domain_pred, torch.zeros(ci_domain_pred.shape).to(self.device)) + \
                              self.criterion_adv(domain_pred, torch.zeros(domain_pred.shape).to(self.device))
            
            loss += task_loss + domain_loss / 3

        loss.backward()

        for opt in opts:
            opt.step()


    def class_disentangle(self):
        updated_models = ['C_R', 'C_IR']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss = 0
        for task in self.x:
            feat = self.netG_D(self.x[task]).detach()
            if task == 'rt_s':
                class_pred = self.netC_R(feat)
            else:
                class_pred = self.netC_IR(feat)

            loss += self.criterion_xent(class_pred, self.y[task]) / 3

        loss.backward()

        for opt in opts:
            opt.step()
        
        updated_models = ['G_D']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss = 0
        for task in self.x:
            feat = self.netG_D(self.x[task])
            if task == 'rt_s':
                class_pred = self.netC_R(feat)
            else:
                class_pred = self.netC_IR(feat)

            loss += - torch.mean(torch.log(torch.nn.functional.softmax(class_pred + 1e-6, dim=-1))) / 3

        loss.backward()

        for opt in opts:
            opt.step()

    
    def collab_learning(self):
        updated_models = ['G_T', 'G_D', 'FS', 'C_R', 'C_IR']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss = 0

        for task in self.x:
            f_ci = self.netG_D(self.x[task])
            f_di = self.netG_T(self.x[task])
            feat = self.netFS(f_ci, f_di)
            
            if task == 'rt_s':
                class_pred = self.netC_R(feat)
            else:
                class_pred = self.netC_IR(feat)
            loss += self.criterion_xent(class_pred, self.y[task]) * self.fs_loss_coeff[task]

        loss.backward()

        for opt in opts:
            opt.step()


    def update_FS(self):
        '''Legacy code'''
        updated_models = ['G_T', 'G_D', 'D_D', 'FS', 'C_R', 'C_IR']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss = 0
        feats = {}
        f_dis = {}
        for task in self.x:
            f_ci = self.netG_D(self.x[task])
            f_di = self.netG_T(self.x[task])
            feat = self.netFS(f_ci, f_di)
            feats[task] = feat
            f_dis[task] = f_di
            
            if task == 'rt_s':
                r_di = f_di.detach()
            elif task == 'irt_t':
                t_ci = f_ci.detach()
            
            
            task_loss = 0
            if task == 'rt_s':
                class_pred = self.netC_R(feat)
                di_class_pred = self.netC_R(f_di)
            else:
                class_pred = self.netC_IR(feat)
                di_class_pred = self.netC_IR(f_di)
            task_loss += self.criterion_xent(class_pred, self.y[task]) * self.fs_loss_coeff[task] + \
                         self.criterion_xent(di_class_pred, self.y[task]) * self.fs_loss_coeff[task] / 3
        

            domain_pred = self.netD_D(self.GRL(f_di))
            ci_domain_pred = self.netD_D(f_ci)
            if task == 'irt_t':
                domain_loss = self.criterion_adv(ci_domain_pred, torch.ones(ci_domain_pred.shape).to(self.device)) + \
                              self.criterion_adv(domain_pred, torch.ones(domain_pred.shape).to(self.device))
            else:
                domain_loss = self.criterion_adv(ci_domain_pred, torch.zeros(ci_domain_pred.shape).to(self.device)) + \
                              self.criterion_adv(domain_pred, torch.zeros(domain_pred.shape).to(self.device))

            loss += task_loss + domain_loss / 3

        loss.backward()

        for opt in opts:
            opt.step()


    def test(self, test_loader, task):
        models = [self.netG_D, self.netG_T, self.netFS, self.netC_R, self.netC_IR]
        for model in models:
            model.eval()

        di_correct = 0
        correct = 0
        count = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                f_ci, f_di = self.netG_D(x), self.netG_T(x)
                if task == 'rt' or task == 'rs':
                    di_pred = self.netC_R(f_di)
                elif task == 'irt' or task == 'irs':
                    di_pred = self.netC_IR(f_di)
                di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y).item()

                feat = self.netFS(f_ci, f_di)

                if task == 'rt' or task == 'rs':
                    pred = self.netC_R(feat)
                elif task == 'irt' or task == 'irs':
                    pred = self.netC_IR(feat)
                correct += torch.sum(torch.argmax(pred, dim=1) == y).item()
                count += len(y)

        di_accuracy = di_correct / count * 100
        accuracy = correct / count * 100

        for model in models:
            model.train()

        return accuracy, di_accuracy

    
    """
    # TSNE plotting

    # import matplotlib
    # matplotlib.use('pdf')
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE

    def tsne_plot(self, test_loaders, stage='', sample_per_class=50):
        models = [self.netG_T, self.netG_D, self.netFS]
        for model in models:
            model.eval()

        rt_classes = self.cfg.rt_classes
        irt_classes = self.cfg.irt_classes

        tsne = TSNE(n_components=2, random_state=0)
        feats = {'rt': [], 'rs': [], 'irt': [], 'irs':[]}
        di_feats = {'rt': [], 'rs': [], 'irt': [], 'irs':[]}
        labels = {'rt': [], 'rs': [], 'irt': [], 'irs':[]}

        with torch.no_grad():
            for task in test_loaders:
                if task in ['rt', 'rs']:
                    max_samples = min(rt_classes * sample_per_class, 750)
                else:
                    max_samples = min(irt_classes * sample_per_class, 750)

                test_loader = test_loaders[task]
                samples = 0
                for x, y in test_loader:
                    if samples >= max_samples:
                        break
                    x = x.to(self.device)
                    y = y.to(self.device)
                    b = x.shape[0]
                    samples += b

                    f_di = self.netG_T(x)
                    f_ci = self.netG_D(x)
                    feat = self.netFS(f_ci, f_di)

                    feats[task].append(feat.cpu().reshape(b, -1))
                    di_feats[task].append(f_di.cpu().reshape(b, -1))
                    labels[task].append(y.cpu())
            
            for task in labels:
                if len(labels[task]) != 0:
                    labels[task] = np.concatenate(labels[task])

            rt_feats = None
            irt_feats = None
            if len(feats['rt']) != 0 or len(feats['rs']) != 0:
                rt_feats = np.concatenate(feats['rt'] + di_feats['rt'] + feats['rs'] + di_feats['rs'])
            if len(feats['irt']) != 0 or len(feats['irs']) != 0:
                irt_feats = np.concatenate(feats['irt'] + di_feats['irt'] + feats['irs'] + di_feats['irs'])

        if rt_feats is not None:
            print('Calculating RT TSNE...')
            rt_feats_2d = tsne.fit_transform(rt_feats)
        if irt_feats is not None:
            print('Calculating IRT TSNE...')
            irt_feats_2d = tsne.fit_transform(irt_feats)
        print('TSNE calculation finished')

        colors = plt.cm.rainbow(np.linspace(0, 1, rt_classes))

        rt_len = 0 if len(labels['rt']) == 0 else labels['rt'].shape[0]
        rs_len = 0 if len(labels['rs']) == 0 else labels['rs'].shape[0]
        irt_len = 0 if len(labels['irt']) == 0 else labels['irt'].shape[0]
        irs_len = 0 if len(labels['irs']) == 0 else labels['irs'].shape[0]

        if rt_len != 0:
            feats_2d = rt_feats_2d[:rt_len]
            di_feats_2d = rt_feats_2d[rt_len:rt_len*2]
            
            plt.figure(figsize=(6, 5))
            for idx, c in enumerate(colors):
                c = np.array([c])
                class_name = str(idx)
                plt.scatter(feats_2d[labels['rt'] == idx, 0], feats_2d[labels['rt'] == idx, 1], marker='o', c=c, label=class_name)
                plt.scatter(di_feats_2d[labels['rt'] == idx, 0], di_feats_2d[labels['rt'] == idx, 1], marker='x', c=c, label=class_name)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, stage + '_rt.png')
            plt.savefig(save_path, dpi=200)
            plt.clf()

        if rs_len != 0:
            feats_2d = rt_feats_2d[rt_len*2:rt_len*2 + rs_len]
            di_feats_2d = rt_feats_2d[rt_len*2 + rs_len:]

            plt.figure(figsize=(6, 5))
            for idx, c in enumerate(colors):
                c = np.array([c])
                class_name = str(idx)
                plt.scatter(feats_2d[labels['rs'] == idx, 0], feats_2d[labels['rs'] == idx, 1], marker='o', c=c, label=class_name)
                plt.scatter(di_feats_2d[labels['rs'] == idx, 0], di_feats_2d[labels['rs'] == idx, 1], marker='x', c=c, label=class_name)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, stage + '_rs.png')
            plt.savefig(save_path, dpi=200)
            plt.clf()

        colors = plt.cm.rainbow(np.linspace(0, 1, irt_classes))

        if irt_len != 0:
            feats_2d = irt_feats_2d[:irt_len]
            di_feats_2d = irt_feats_2d[irt_len:irt_len*2]

            plt.figure(figsize=(6, 5))
            for idx, c in enumerate(colors):
                c = np.array([c])
                class_name = str(idx)
                plt.scatter(feats_2d[labels['irt'] == idx, 0], feats_2d[labels['irt'] == idx, 1], marker='o', c=c, label=class_name)
                plt.scatter(di_feats_2d[labels['irt'] == idx, 0], di_feats_2d[labels['irt'] == idx, 1], marker='x', c=c, label=class_name)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, stage + '_irt.png')
            plt.savefig(save_path, dpi=200)
            plt.clf()
        
        if irs_len != 0:
            feats_2d = irt_feats_2d[irt_len*2:irt_len*2 + irs_len]
            di_feats_2d = irt_feats_2d[irt_len*2 + irs_len:]

            plt.figure(figsize=(6, 5))
            for idx, c in enumerate(colors):
                c = np.array([c])
                class_name = str(idx)
                plt.scatter(feats_2d[labels['irs'] == idx, 0], feats_2d[labels['irs'] == idx, 1], marker='o', c=c, label=class_name)
                plt.scatter(di_feats_2d[labels['irs'] == idx, 0], di_feats_2d[labels['irs'] == idx, 1], marker='x', c=c, label=class_name)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, stage + '_irs.png')
            plt.savefig(save_path, dpi=200)
            plt.clf()

        for model in models:
            model.train()
    """

    def get_current_loss(self):
        losses = {}
        for k in self.losses:
            losses[k] = self.losses[k] / self.counts[k]
        return losses

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.device != torch.device('cpu'):
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if name == 'D_D':
                continue
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)

            if not os.path.exists(load_path):
                print(load_path, "not exists")
                continue

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.cfg.lr_policy == 'plateau':
                pass
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)