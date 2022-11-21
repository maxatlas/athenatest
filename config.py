"""
Define hyper-parameters here.

"""

accuracy_threshold = 0.5

res_folder_path = "results"
test_fp_folder = "testing/results/false_positives"
fp_folder_path = "%s/false_positives" % res_folder_path

batch_size_test = 100
random_seed = 122

k = 10

model = "tiny"
data_folder_path = "../data/cifar10/test"
test_plot_folder = "vis/"
shuffle_dataloader = True

pad = 2

device = "cuda:0"
save_fp = False

testing_limit = 3  # Use 3 samples for testing
