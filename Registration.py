import argparse
import os
import time
from Network import BrainNet
from Loss import *
from NeuralODE import *
from Utils import *

def main(config):
    device = torch.device(config.device)
    fixed = load_nii(config.fixed)
    moving = load_nii(config.moving)
    assert fixed.shape == moving.shape  # two images to be registered must in the same size
    t = time.time()
    df, df_with_grid, warped_moving = registration(config, device, moving, fixed)
    runtime = time.time() - t
    print('Registration Running Time:', runtime)
    print('---Registration DONE---')
    evaluation(config, device, df, df_with_grid)
    print('---Evaluation DONE---')
    save_result(config, df, warped_moving)
    print('---Results Saved---')


def registration(config, device, moving, fixed):
    '''
    Registration moving to fixed.
    :param config: configurations.
    :param device: gpu or cpu.
    :param img1: moving image to be registered, geodesic shooting starting point.
    :param img2: fixed image, geodesic shooting target.
    :return ode_train: neuralODE class.
    :return all_phi: Displacement field for all time steps.
    '''
    im_shape = fixed.shape
    moving = torch.from_numpy(moving).to(device).float()
    fixed = torch.from_numpy(fixed).to(device).float()
    # make batch dimension
    moving = moving.unsqueeze(0).unsqueeze(0)
    fixed = fixed.unsqueeze(0).unsqueeze(0)

    Network = BrainNet(img_sz=im_shape,
                       smoothing_kernel=config.smoothing_kernel,
                       smoothing_win=config.smoothing_win,
                       smoothing_pass=config.smoothing_pass,
                       ds=config.ds,
                       bs=config.bs
                       ).to(device)

    ode_train = NeuralODE(Network, config.optimizer, config.STEP_SIZE).to(device)

    # training loop
    scale_factor = torch.tensor(im_shape).to(device).view(1, 3, 1, 1, 1) * 1.
    ST = SpatialTransformer(im_shape).to(device)  # spatial transformer to warp image
    grid = generate_grid3D_tensor(im_shape).unsqueeze(0).to(device)  # [-1,1]

    # Define optimizer
    optimizer = torch.optim.Adam(ode_train.parameters(), lr=config.lr, amsgrad=True)
    loss_NCC = NCC(win=config.NCC_win)
    BEST_loss_sim_loss_J = 1000
    for i in range(config.epoches):
        all_phi = ode_train(grid, Tensor(np.arange(config.time_steps)), return_whole_sequence=True)
        all_v = all_phi[1:] - all_phi[:-1]
        all_phi = (all_phi + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing
        phi = all_phi[-1]
        grid_voxel = (grid + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing
        df = phi - grid_voxel  # with grid -> without grid
        warped_moving, df_with_grid = ST(moving, df, return_phi=True)
        # similarity loss
        loss_sim = loss_NCC(warped_moving, fixed)
        warped_moving = warped_moving.squeeze(0).squeeze(0)
        # V magnitude loss
        loss_v = config.lambda_v * magnitude_loss(all_v)
        # neg Jacobian loss
        loss_J = config.lambda_J * neg_Jdet_loss(df_with_grid)
        # phi dphi/dx loss
        loss_df = config.lambda_df * smoothloss_loss(df)
        loss = loss_sim + loss_v + loss_J + loss_df
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print("Iteration: {0} Loss_sim: {1:.3e} loss_J: {2:.3e}".format(i + 1, loss_sim.item(), loss_J.item()))
        # pick the one df with the most balance loss_sim and loss_J in the last 50 epoches
        if i > config.epoches - 50:
            loss_sim_loss_J = 1000 * loss_sim.item() * loss_J.item()
            if loss_sim_loss_J < BEST_loss_sim_loss_J:
                best_df = df.detach().clone()
                best_df_with_grid = df_with_grid.detach().clone()
                best_warped_moving = warped_moving.detach().clone()
    return best_df, best_df_with_grid, best_warped_moving


def evaluation(config, device, df, df_with_grid):
    ### Calculate Neg Jac Ratio
    neg_Jet = -1.0 * JacboianDet(df_with_grid)
    neg_Jet = F.relu(neg_Jet)
    mean_neg_J = torch.sum(neg_Jet).detach().cpu().numpy()
    num_neg = len(torch.where(neg_Jet > 0)[0])
    total = neg_Jet.size(-1) * neg_Jet.size(-2) * neg_Jet.size(-3)
    ratio_neg_J = num_neg / total
    print('Total of neg Jet: ', mean_neg_J)
    print('Ratio of neg Jet: ', ratio_neg_J)
    ### Calculate Dice
    label = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
    fixed_seg = load_nii(config.fixed_seg)
    moving_seg = load_nii(config.moving_seg)
    ST_seg = SpatialTransformer(fixed_seg.shape, mode='nearest').to(device)
    moving_seg = torch.from_numpy(moving_seg).to(device).float()
    # make batch dimension
    moving_seg = moving_seg[None, None, ...]
    warped_seg = ST_seg(moving_seg, df, return_phi=False)
    dice_move2fix = dice(warped_seg.unsqueeze(0).unsqueeze(0).detach().cpu().numpy(), fixed_seg, label)
    print('Avg. dice on %d structures: ' % len(label), np.mean(dice_move2fix[0]))

def save_result(config, df, warped_moving):
    save_nii(df.permute(2,3,4,0,1).detach().cpu().numpy(), '%s/df.nii.gz' % (config.savepath))
    save_nii(warped_moving.detach().cpu().numpy(), '%s/warped.nii.gz' % (config.savepath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # File path
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='./result',
                        help="path for saving results")
    parser.add_argument("--fixed", type=str,
                        dest="fixed", default='./data/OAS1_0001_MR1/brain.nii.gz',
                        help="fixed image data path")
    parser.add_argument("--moving", type=str,
                        dest="moving", default='./data/OAS1_0002_MR1/brain.nii.gz',
                        help="moving image data path")
    parser.add_argument("--fixed_seg", type=str,
                        dest="fixed_seg", default='./data/OAS1_0001_MR1/brain_aseg.nii.gz',
                        help="fixed image segmentation data path")
    parser.add_argument("--moving_seg", type=str,
                        dest="moving_seg", default='./data/OAS1_0002_MR1/brain_aseg.nii.gz',
                        help="moving image segmentation data path")
    # Model configuration
    parser.add_argument("--ds", type=int,
                        dest="ds", default=2,
                        help="specify output downsample times.")
    parser.add_argument("--bs", type=int,
                        dest="bs", default=16,
                        help="bottleneck size.")
    parser.add_argument("--smoothing_kernel", type=str,
                        dest="smoothing_kernel", default='AK',
                        help="AK: Averaging kernel; GK: Gaussian Kernel")
    parser.add_argument("--smoothing_win", type=int,
                        dest="smoothing_win", default=15,
                        help="Smoothing Kernel size")
    parser.add_argument("--smoothing_pass", type=int,
                        dest="smoothing_pass", default=1,
                        help="Number of Smoothing pass")
    # Training configuration
    parser.add_argument("--time_steps", type=int,
                        dest="time_steps", default=2,
                        help="number of time steps between the two images, >=2.")
    parser.add_argument("--optimizer", type=str,
                        dest="optimizer", default='Euler',
                        help="Euler or RK.")
    parser.add_argument("--STEP_SIZE", type=float,
                        dest="STEP_SIZE", default=0.001,
                        help="step size for numerical integration.")
    parser.add_argument("--epoches", type=int,
                        dest="epoches", default=300,
                        help="No. of epochs to train.")
    parser.add_argument("--NCC_win", type=int,
                        dest="NCC_win", default=21,
                        help="NCC window size")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.005,
                        help="Learning rate.")
    parser.add_argument("--lambda_J", type=int,
                        dest="lambda_J", default=2.5,
                        help="Loss weight for neg J")
    parser.add_argument("--lambda_df", type=int,
                        dest="lambda_df", default=0.05,
                        help="Loss weight for dphi/dx")
    parser.add_argument("--lambda_v", type=int,
                        dest="lambda_v", default=0.00005,
                        help="Loss weight for neg J")
    parser.add_argument("--loss_sim", type=str,
                        dest="loss_sim", default='NCC',
                        help="Similarity measurement")
    # Debug
    parser.add_argument("--debug", type=bool,
                        dest="debug", default=False,
                        help="debug mode")
    # Device
    parser.add_argument("--device", type=str,
                        dest="device", default='cuda:0',
                        help="gpu: cuda:0; cpu: cpu")

    config = parser.parse_args()
    if not os.path.isdir(config.savepath):
        os.makedirs(config.savepath)
    main(config)
