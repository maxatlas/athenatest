"""
Define hyper-parameters here.

"""

delimiter = "+"
min_n = 20

deviation_threshold = 0.1

res_folder_path = "../results"
fp_folder_path = "%s/false_positives" % res_folder_path

mce_table_name = "MCE"
ece_table_name = "ECE"
cm_table_name = "confusion_matrix"

batch_size_test = 100

random_seed = 122

k = 10
data_folder_path = "../data/mnist/test"
test_output_folder = "vis/"
shuffle_dataloader = True

pad = 2
decimal_precision = 5

