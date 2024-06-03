import os
import sys
import time
import yaml
# select the given cuda device
# os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"
import torch
from torch.utils.data.dataloader import DataLoader
from utils import new_data_loader
from utils import model_factory

cuda_avail = torch.cuda.is_available()
# start_time = time.time()

def test_net(params):
    # Determine whether to use GPU
    if params['use_gpu'] == 1:
        print("GPU:" + str(params['use_gpu']))

    if params['use_gpu'] == 1 and cuda_avail:
        print("use_gpu=True and Cuda Available. Setting Device=CUDA")
        device = torch.device("cuda:0")  # change the GPU index based on the availability
        use_gpu = True
    else:
        print("Setting Device=CPU")
        device = torch.device("cpu")
        use_gpu = False


    # Set seed
    if params['use_random_seed'] == 0:
        torch.manual_seed(params['seed'])

    # Create network & Init Layer weights
    if params['Modality'] == "Combined":
        NN_model, model_params = model_factory.get_model(params, use_gpu)
    # Focus on using two sensor inputs
    elif params['Modality'] == "Tactile" or params['Modality'] == "Visual":
        NN_model, model_params = model_factory_single.get_model(params, use_gpu)


    if use_gpu:
        NN_model = NN_model.cuda()

        model_path = "Trained_Model/timeSformer_orig_two/30_05_2023__16_21_11/timeSformer_orig_two_last.pt"
        NN_model = torch.load(model_path)
        NN_model.eval()  # set the model to evaluation mode if needed NN_model.cuda() # move the model to GPU if needed

    # 加载数据集   Dataloader
    test_dataset = new_data_loader.Tactile_Vision_dataset(params['Fruit_type'], params['label_encoding'],params["Tactile_scale_ratio"], params["Visual_scale_ratio"], params["video_length"],
                                                      data_path=params['Test_data_dir'])
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True,
                                  num_workers=params['num_workers'])


    # test_acc = []
    # Start testing
    start_time = time.time()
    # Start
    test_total_acc = 0.0
    test_total = 0.0
    test_acc = 0
    for i, data in enumerate(test_data_loader):
        NN_model.zero_grad()
        label = data[2]
        if params['Modality'] == "Combined":
            output = NN_model(data[0], data[1], data[3])
        # Focus on two sensor input case
        elif params['Modality'] == "Visual":
            output = NN_model(data[0], data[3])
        elif params['Modality'] == "Tactile":
            output = NN_model(data[1], data[3])
        if use_gpu:
            label = label.to('cuda')
        # cal testing acc
        _, predicted = torch.max(output.data, 1)
        test_total_acc += (predicted == label).sum().item()
        test_total += len(label)
        test_acc = (test_total_acc / test_total)
    print('Test Accuracy of the model on the {} test images: {}% '.format(test_total, test_acc * 100))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Testing time: {:.2f} seconds'.format(elapsed_time))


    # message = "Elapsed {:.2f}s, {:.2f} s/batch, ets {:.2f}s".format(
    #     elapsed_time, speed_batch, eta)
    # print(message)
    # log_record.update_log(exp_save_dir, message)

if __name__ == "__main__":
    exp_name = 'config_timeSformer.yaml' # make sure config.yaml is under the same directory

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]  #or we can explicitly specify the file name

    print("Running Experiment: ", exp_name)
    yaml_file = exp_name
    if os.path.exists(yaml_file):
        with open(yaml_file) as stream:
            config_loaded = yaml.safe_load(stream)
    else:
        print("The yaml file does not exist!")
        sys.exit()
    test_net(config_loaded) # test the model, with the specifications

