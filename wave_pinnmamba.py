import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS,Adam
from tqdm import tqdm
import os
import argparse
from util import *
from model_dict import get_model

#torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = False

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
step_size = 1e-2
num_step = 7
seq_diff = int(1e-2/step_size)

parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='pinn')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

res, b_left, b_right, b_upper, b_lower = get_data([0, 1], [0, 1], 101, 101)
res_test, _, _, _, _ = get_data([0, 1], [0, 1], 101, 101)

if args.model == 'PINNsFormer' or args.model == 'PINNMamba':
    res = make_time_sequence(res, num_step=num_step, step=step_size)
    b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
    b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
    b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
    b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)
    
if args.model == 'PINNsFormer' or args.model == 'PINNMamba':
    res_test = make_time_sequence(res_test, num_step=num_step, step=step_size)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, \
                                  noise_scale_base=0.25, device=device).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)
elif args.model == 'PINNsFormer':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    model.apply(init_weights)
elif args.model == 'PINNMamba':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)

optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

n_params = get_n_params(model)

print(model)
print(get_n_params(model))

loss_track = []
w1, w2, w3 = 1, 1, 1
pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)

kernel_size = 300

D1 = kernel_size
D2 = len(x_left)
D3 = len(x_lower)

def compute_ntk(J1, J2):
    Ker = torch.matmul(J1, torch.transpose(J2, 0, 1))
    return Ker

'''  
    if i % 20 == 0:
        J1 = torch.zeros((D1, n_params))
        J2 = torch.zeros((D2, n_params))
        J3 = torch.zeros((D3, n_params))

        batch_ind = np.random.choice(len(x_res), kernel_size, replace=False)
        x_train, t_train = x_res[batch_ind], t_res[batch_ind]

        pred_res = model(x_train, t_train)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        for j in range(len(x_train)):
            model.zero_grad()
            pred_res[j,0].backward(retain_graph=True)
            J1[j, :] = torch.cat([p.grad.view(-1) for p in model.parameters()])

        for j in range(len(x_left)):
            model.zero_grad()
            pred_left[j,0].backward(retain_graph=True)
            J2[j, :] = torch.cat([p.grad.view(-1) for p in model.parameters()])

        for j in range(len(x_lower)):
            model.zero_grad()
            pred_lower[j,0].backward(retain_graph=True)
            pred_upper[j,0].backward(retain_graph=True)
            J3[j, :] = torch.cat([p.grad.view(-1) for p in model.parameters()])

        K1 = torch.trace(compute_ntk(J1, J1))
        K2 = torch.trace(compute_ntk(J2, J2))
        K3 = torch.trace(compute_ntk(J3, J3))
        
        K = K1+K2+K3

        w1 = K.item() / K1.item()
        w2 = K.item() / K2.item()
        w3 = K.item() / K3.item()
'''   
for i in tqdm(range(1000)):  
    def closure():


        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        #pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_tt - 4 * u_xx) ** 2)
        loss_bc = torch.mean((pred_upper) ** 2) + torch.mean((pred_lower) ** 2)

        ui_t = torch.autograd.grad(pred_left, t_left, grad_outputs=torch.ones_like(pred_left), retain_graph=True, create_graph=True)[0]

        loss_ic_1 = torch.mean((pred_left[:,0] - torch.sin(pi*x_left[:,0]) - 0.5 * torch.sin(3*pi*x_left[:,0])) ** 2)
        loss_ic_2 = torch.mean((ui_t)**2)

        loss_ic = loss_ic_1 + loss_ic_2

        loss_seq = torch.mean((pred_res[101:10201,0:num_step-seq_diff,:]-pred_res[0:10100,seq_diff:num_step,:]) ** 2)


        loss_track.append([loss_res.item(), loss_ic_1.item(), loss_ic_2.item(), loss_bc.item(), loss_seq.item()])

        #print('Loss Res: {:4f}, Loss_IC1: {:4f} ,Loss_IC2: {:4f}, Loss_BC: {:4f}, Loss_SQ: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2] ,loss_track[-1][3] ,loss_track[-1][4]))
        
        loss = w1 * loss_res + w2 * loss_ic +  w3 * loss_bc +  0.1 * w1 / num_step * loss_seq
        #loss = loss_res +  loss_ic + loss_bc # + 100 * loss_seq
        optim.zero_grad()
        loss.backward()
        return loss


    optim.step(closure)

    if i % 20 == 0:
        print('Loss Res: {:4f}, Loss_IC1: {:4f} ,Loss_IC2: {:4f}, Loss_BC: {:4f}, Loss_SQ: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2] ,loss_track[-1][3] ,loss_track[-1][4]))
       
#model.load_state_dict(torch.load("./results/1dwave_PINNMamba_7_0.01_real_sequence_point.pt"))
        

#print('Loss Res: {:4f}, Loss_IC: {:4f}, Loss_BC: {:4f}, Loss_SQ: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2],loss_track[-1][3]))
#print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

if not os.path.exists('./results/wave/'):
    os.makedirs('./results/wave/')

torch.save(model.state_dict(), f'./results/wave/1dwave_{args.model}_{num_step}_{step_size}.pt')

# Visualize
#if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
   # res_test = make_time_sequence(res_test, num_step=5, step=1e-4)

res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

with torch.no_grad():
    pred = model(x_test, t_test)[:, 0:1]
    pred = pred.cpu().detach().numpy()

pred = pred.reshape(101, 101)


def u_ana(x, t):
    return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)


res_test, _, _, _, _ = get_data([0, 1], [0, 1], 101, 101)
u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

plt.figure(figsize=(4, 3))
plt.imshow(pred, extent=[0,1,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig(f'./results/wave/1dwave_{args.model}_{num_step}_{step_size}_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u, extent=[0,1,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig('./results/wave/1dwave_exact.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, extent=[0,1,1,0], aspect='auto', cmap='coolwarm', vmin=-0.3, vmax=0.3)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig(f'./results/wave/1dwave_{args.model}_{num_step}_{step_size}_error.pdf', bbox_inches='tight')


