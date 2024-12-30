import os

# create train, test data
os.system('copy npy\\actor.npy train\\actor_train.npy')
os.system('copy npy\\nonactor.npy train\\nonactor_train.npy')
os.system('copy npy\\test.npy test\\actor_test.npy')
os.system('copy npy\\test.npy test\\nonactor_test.npy')

# create train, test label
os.system('copy npy\\actor_label.npy train\\actor_train_label.npy')
os.system('copy npy\\nonactor_label.npy train\\nonactor_train_label.npy')
os.system('copy npy\\test_label.npy test\\actor_test_label.npy')
os.system('copy npy\\test_label.npy test\\nonactor_test_label.npy')