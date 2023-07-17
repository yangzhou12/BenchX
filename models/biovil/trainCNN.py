import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from utils import *
import sys
sys.path.insert(0, '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/src/evaluate/')
from os.path import exists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

elapsed = time.time() - t
print("Start (time = " + str(elapsed) + ")")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    t = time.time()
    # Start experiment

    mod_path = args.model_path + args.model + "/"
    exp_path = getExperiment(args, mod_path)
    start, cnn_model, params, optimizer, best_val_loss = startExperiment(args, exp_path, cnn=True, pretrained=False)

    filts = 'frontal'
    if exists(exp_path + '/filters.txt'):
        filters = MedDataHelpers.getFilters(exp_path)
        if set(filters) != set(MedDataHelpers.getFilters(exp_path, overwrite= filts, toprint=False)):
            raise Exception("Error: entered filters differ from those previously used")
    else:
        filters = MedDataHelpers.getFilters(exp_path, overwrite = filts)
        if exp_path != 'debug':
            with open(exp_path + '/filters.txt', 'w') as f:
                f.write(filts)
    # Build data
    if args.debug:
        subset = ['tinytrain', 'tinyval']
    else:
        subset = ['train', 'val']
    t, v = subset[0], subset[1]

    mimic_dat = MedDataHelpers.getDatasets(source='m', subset = subset, augs = 1, filters = filters)
    dls = MedDataHelpers.getLoaders(mimic_dat, args)
    train_data_loader_mimic, val_data_loader_mimic = dls[t], dls[v]
    total_step_mimic = len(train_data_loader_mimic)
    assert (args.resume or start == 0)
    # Train and validate

    for epoch in range(start, args.num_epochs):
        cnn_model.train()
        tmimic = time.time()
        train_loss = train_vision(train_data_loader_mimic, cnn_model, args, epoch, optimizer, total_step_mimic, heads)

        print("Mimic Epoch time: " + str(time.time() - tmimic))
        if epoch % args.val_step == 0:
            print("Validating/saving model")
            cnn_model.eval()
            tval = time.time()
            val_loss = validate_vision(val_data_loader_mimic, cnn_model, args, heads)
            if not args.debug:
                torch.save({'epoch': epoch+1,
                            'model_state_dict': cnn_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'args': args}, os.path.join(exp_path, 'je_model-{}.pt'.format(epoch)))
                if val_loss <= best_val_loss:
                    print("Best model so far!")
                    best_val_loss = val_loss
                    torch.save({'epoch': epoch + 1,
                                'model_state_dict': cnn_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss':best_val_loss,
                                'val_loss': val_loss,
                                'args': args}, os.path.join(exp_path, 'best_model.pt'))


            print("Val time " + str(time.time() - tval))

#For baseline supervised CNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model information
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='cxr_cnn')
    parser.add_argument('--resume', type=int, default=0, const=-1, nargs='?')
    parser.add_argument('--debug', type=bool, default=False, const=True, nargs='?', help='debug mode, dont save')

    #Training parameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    parser.add_argument('--log_step', type=int, default=500, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=2, help='step size for printing val info')
    args = parser.parse_args()
    print(args)
    main(args)