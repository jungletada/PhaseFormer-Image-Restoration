import os
import torch


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_model(opt, epoch, name):
    load_epoch = '/' + str(epoch) + '-'
    return opt.save_models_dir + load_epoch + name


def load_model_ckpt(model, epoch, opt, name):
    ckpt = torch.load(get_model(opt, epoch, name))
    model.load_state_dict(ckpt['model_state_dict'])


def save_model_ckpt(model, epoch, opt, name):
    state = {'model_state_dict': model.state_dict()}
    torch.save(state, get_model(opt, epoch, name))


def load_net(model, model_path):
    model.load_state_dict(torch.load(model_path), strict=True)


def save_net(model, save_path):
    torch.save(model.state_dict(), save_path)