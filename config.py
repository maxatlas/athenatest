"""
Define hyper-parameters here.

"""

accuracy_threshold = 0
model = "tiny"
data_folder_path = "../data/cifar10/test"
device = "cuda:0"
save_fp = True


res_folder_path = "results"
test_fp_folder = "testing/results/false_positives"
fp_folder_path = "%s/false_positives" % res_folder_path
test_plot_folder = "vis/"

batch_size_test = 100  # batch size for test set
random_seed = 122
k = 10  # Bin size
shuffle_dataloader = True

pad = 2  # image padding. pad < 2 will break MNIST eval
testing_limit = 3  # Use 3 samples for fp save test

n_cluster = 256  # Kmeans cluster
