import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import os
import json
import itertools

# Define the model suffix (change this to test different models)
isModifDoss = "Original_code_DQN_E30D3A1"
isModifDataset = "original_DQN_E30D3A1"

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Define huber_loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)

# Register huber_loss for Keras
import tensorflow.keras.losses
tensorflow.keras.losses.huber_loss = huber_loss

# Fast Gradient Sign Method (FGSM) for adversarial examples
def generate_adversarial_examples(model, x, y, epsilon=0.1):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x)
        loss = huber_loss(y, predictions)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    adversarial_x = x + epsilon * signed_grad
    adversarial_x = tf.clip_by_value(adversarial_x, x - epsilon, x + epsilon)
    return adversarial_x.numpy()

# RLenv class
class RLenv:
    def __init__(self, train_test, **kwargs):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
                     "dst_bytes","land_f","wrong_fragment","urgent","hot","num_failed_logins",
                     "logged_in","num_compromised","root_shell","su_attempted","num_root",
                     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                     "is_host_login","is_guest_login","count","srv_count","serror_rate",
                     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                     "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                     "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","dificulty"]
        self.index = 0
        self.loaded = False
        self.train_test = train_test
        self.formated_train_path = kwargs.get('formated_train_path', f"formated_train_adv_{isModifDataset}.data")
        self.formated_test_path = kwargs.get('formated_test_path', f"formated_test_adv_{isModifDataset}.data")
        self.attack_types = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
        self.attack_names = []
        self.attack_map = {
            'normal': 'normal', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
            'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
            'processtable': 'DoS', 'udpstorm': 'DoS', 'ipsweep': 'Probe', 'nmap': 'Probe',
            'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
            'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
            'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L', 'buffer_overflow': 'U2R',
            'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'httptunnel': 'U2R',
            'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
        }
        self.all_attack_names = list(self.attack_map.keys())

    def get_shape(self):
        if not self.loaded:
            self._load_df()
        self.data_shape = self.df.shape
        return self.data_shape

    def get_full(self):
        if not self.loaded:
            self._load_df()
        labels = self.df[self.attack_names]
        batch = self.df.drop(self.all_attack_names, axis=1)
        return batch, labels

    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path, sep=',')
        else:
            self.df = pd.read_csv(self.formated_test_path, sep=',')
        self.index = np.random.randint(0, self.df.shape[0]-1, dtype=np.int32)
        self.loaded = True
        for att in self.attack_map:
            if att in self.df.columns and np.sum(self.df[att].values) > 1:
                self.attack_names.append(att)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Normalized confusion matrix - {title}")
    else:
        print(f'Confusion matrix, without normalization - {title}')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Evaluate multiclass subset
def evaluate_multiclass_subset(states, true_labels, detailed_labels, predicted_actions, label_subset, subset_name, attack_types, results_dir, isModifDoss, use_detailed=False):
    subset_indices = [i for i, d_label in enumerate(detailed_labels) if d_label in label_subset] if use_detailed else range(len(true_labels))
    
    if not subset_indices:
        print(f"No samples found for {subset_name}")
        return None, None, None, None

    subset_true_labels = true_labels[subset_indices]
    subset_predicted = predicted_actions[subset_indices]

    total_reward = np.sum(subset_true_labels == subset_predicted)
    num_samples = len(subset_true_labels)
    acc = float(100 * total_reward / num_samples) if num_samples > 0 else 0.0

    f1 = f1_score(subset_true_labels, subset_predicted, average='weighted')
    prec = precision_score(subset_true_labels, subset_predicted, average='weighted')
    rec = recall_score(subset_true_labels, subset_predicted, average='weighted')

    estimated_labels = np.bincount(subset_predicted, minlength=len(attack_types))
    true_labels_count = np.bincount(subset_true_labels, minlength=len(attack_types))
    correct_labels = np.zeros(len(attack_types), dtype=int)
    for i, (true, pred) in enumerate(zip(subset_true_labels, subset_predicted)):
        if true == pred:
            correct_labels[true] += 1
    mismatch = estimated_labels - true_labels_count

    outputs_df = pd.DataFrame(index=attack_types, columns=["Estimated", "Correct", "Total", "F1_score", "Mismatch"])
    for i, cls in enumerate(attack_types):
        outputs_df.iloc[i].Estimated = estimated_labels[i]
        outputs_df.iloc[i].Correct = correct_labels[i]
        outputs_df.iloc[i].Total = true_labels_count[i]
        outputs_df.iloc[i].F1_score = f1_score(subset_true_labels, subset_predicted, labels=[i], average=None)[0] * 100 if true_labels_count[i] > 0 else 0.0
        outputs_df.iloc[i].Mismatch = abs(mismatch[i])

    print(f'\n{subset_name} - Total reward: {total_reward} | Number of samples: {num_samples} | Accuracy = {acc:.2f}%')
    print(outputs_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    pos = np.arange(len(true_labels_count))
    plt.bar(pos, correct_labels, width, color='g', label='Correct estimated')
    plt.bar(pos + width, np.abs(correct_labels - true_labels_count), width, color='r', label='False negative')
    plt.bar(pos + width, np.abs(estimated_labels - correct_labels), width,
            bottom=np.abs(correct_labels - true_labels_count), color='b', label='False positive')
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(attack_types, rotation='vertical', fontsize='xx-large')
    ax.yaxis.set_tick_params(labelsize=15)
    plt.legend(('Correct estimated', 'False negative', 'False positive'), fontsize='x-large')
    plt.title(f"{subset_name} Performance")
    plt.tight_layout()
    bar_chart_path = f'{results_dir}/test_bar_{subset_name.lower().replace(" ", "_")}_{isModifDoss}.svg'
    plt.savefig(bar_chart_path, format='svg', dpi=1000)
    plt.close()
    print(f"Bar chart saved to: {bar_chart_path}")

    cnf_matrix = confusion_matrix(subset_true_labels, subset_predicted)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title=f'Normalized Confusion Matrix - {subset_name}')
    cm_plot_path = f'{results_dir}/test_confusion_matrix_{subset_name.lower().replace(" ", "_")}_{isModifDoss}.svg'
    plt.savefig(cm_plot_path, format='svg', dpi=1000)
    plt.close()
    print(f"Confusion matrix saved to: {cm_plot_path}")

    return acc, f1, prec, rec

# Binary classification evaluation
def evaluate_binary_subset(states, true_labels, detailed_labels, predicted_actions, exclusive_labels, subset_name, attack_types, results_dir, isModifDoss):
    binary_true_labels = np.array([0 if attack_types[label] == 'normal' else 1 for label in true_labels])
    subset_indices = [i for i, d_label in enumerate(detailed_labels) if d_label in exclusive_labels]
    
    if not subset_indices:
        print(f"No samples found for {subset_name}")
        return None, None, None, None

    subset_states = states.iloc[subset_indices]
    subset_true_labels = binary_true_labels[subset_indices]
    subset_predicted = np.array([0 if attack_types[pred] == 'normal' else 1 for pred in predicted_actions[subset_indices]])

    total_reward = np.sum(subset_true_labels == subset_predicted)
    num_samples = len(subset_true_labels)
    acc = float(100 * total_reward / num_samples) if num_samples > 0 else 0.0

    binary_classes = ['Normal', 'Attack']
    f1 = f1_score(subset_true_labels, subset_predicted, average='weighted')
    prec = precision_score(subset_true_labels, subset_predicted, average='weighted')
    rec = recall_score(subset_true_labels, subset_predicted, average='weighted')

    estimated_labels = np.bincount(subset_predicted, minlength=2)
    true_labels_count = np.bincount(subset_true_labels, minlength=2)
    correct_labels = np.zeros(2, dtype=int)
    for i, (true, pred) in enumerate(zip(subset_true_labels, subset_predicted)):
        if true == pred:
            correct_labels[true] += 1
    mismatch = estimated_labels - true_labels_count

    outputs_df = pd.DataFrame(index=binary_classes, columns=["Estimated", "Correct", "Total", "F1_score", "Mismatch"])
    for i, cls in enumerate(binary_classes):
        outputs_df.iloc[i].Estimated = estimated_labels[i]
        outputs_df.iloc[i].Correct = correct_labels[i]
        outputs_df.iloc[i].Total = true_labels_count[i]
        outputs_df.iloc[i].F1_score = f1_score(subset_true_labels, subset_predicted, labels=[i], average=None)[0] * 100 if true_labels_count[i] > 0 else 0.0
        outputs_df.iloc[i].Mismatch = abs(mismatch[i])

    print(f'\n{subset_name} - Total reward: {total_reward} | Number of samples: {num_samples} | Accuracy = {acc:.2f}%')
    print(outputs_df)

    fig, ax = plt.subplots(figsize=(6, 6))
    width = 0.35
    pos = np.arange(len(true_labels_count))
    plt.bar(pos, correct_labels, width, color='g', label="Correct estimated")
    plt.bar(pos + width, np.abs(correct_labels - true_labels_count), width, color='r', label="False negative")
    plt.bar(pos + width, np.abs(estimated_labels - correct_labels), width, 
            bottom=np.abs(correct_labels - true_labels_count), color='b', label="False positive")
    ax.set_xticks(pos + width / 2)
    ax.set_xticklabels(binary_classes, fontsize='xx-large')
    ax.yaxis.set_tick_params(labelsize=15)
    plt.legend(fontsize='x-large')
    plt.title(f"{subset_name} Performance (Binary)")
    plt.tight_layout()
    bar_chart_path = f'{results_dir}/test_bar_binary_{subset_name.lower().replace(" ", "_")}_{isModifDoss}.svg'
    plt.savefig(bar_chart_path, format='svg', dpi=1000)
    plt.close()
    print(f"Binary bar chart saved to: {bar_chart_path}")

    cnf_matrix = confusion_matrix(subset_true_labels, subset_predicted)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=binary_classes, normalize=True, title=f'Normalized Confusion Matrix - {subset_name} (Binary)')
    cm_plot_path = f'{results_dir}/test_confusion_matrix_binary_{subset_name.lower().replace(" ", "_")}_{isModifDoss}.svg'
    plt.savefig(cm_plot_path, format='svg', dpi=1000)
    plt.close()
    print(f"Binary confusion matrix saved to: {cm_plot_path}")

    return acc, f1, prec, rec

# Function to evaluate a model and return metrics
def evaluate_model(model, test_states, test_labels, test_detailed_labels, env_test, results_dir, isModifDoss, model_name):
    # Predict actions
    q = model.predict(test_states)
    actions = np.argmax(q, axis=1)

    # Evaluate All Labels (multiclass)
    acc_all, f1_all, prec_all, rec_all = evaluate_multiclass_subset(
        test_states, test_mapped_labels, test_detailed_labels, actions,
        set(env_test.attack_types), f"All Labels ({model_name})", env_test.attack_types, results_dir, isModifDoss, use_detailed=False
    )

    # Evaluate Common Labels (multiclass)
    acc_common = f1_common = prec_common = rec_common = None
    if common_labels:
        acc_common, f1_common, prec_common, rec_common = evaluate_multiclass_subset(
            test_states, test_mapped_labels, test_detailed_labels, actions,
            common_labels, f"Common Labels ({model_name})", env_test.attack_types, results_dir, isModifDoss, use_detailed=True
        )

    # Evaluate Exclusive Labels (multiclass)
    acc_exclusive = f1_exclusive = prec_exclusive = rec_exclusive = None
    if exclusive_labels:
        acc_exclusive, f1_exclusive, prec_exclusive, rec_exclusive = evaluate_multiclass_subset(
            test_states, test_mapped_labels, test_detailed_labels, actions,
            exclusive_labels, f"Exclusive Labels ({model_name})", env_test.attack_types, results_dir, isModifDoss, use_detailed=True
        )

    # Evaluate Exclusive Binary (Normal vs Attack)
    acc_binary, f1_binary, prec_binary, rec_binary = evaluate_binary_subset(
        test_states, test_mapped_labels, test_detailed_labels, actions,
        exclusive_labels, f"Exclusive Binary (Normal vs Attack) ({model_name})", env_test.attack_types, results_dir, isModifDoss
    )

    # Print summary of metrics
    print(f"\nSummary of Metrics for {isModifDoss} ({model_name}):")
    print(f"All Labels:")
    print(f"  Accuracy: {acc_all:.2f}%")
    print(f"  F1-Score: {f1_all * 100:.2f}%")
    print(f"  Precision: {prec_all * 100:.2f}%")
    print(f"  Recall: {rec_all * 100:.2f}%")
    
    print(f"\nCommon Labels:")
    if acc_common is not None:
        print(f"  Accuracy: {acc_common:.2f}%")
        print(f"  F1-Score: {f1_common * 100:.2f}%")
        print(f"  Precision: {prec_common * 100:.2f}%")
        print(f"  Recall: {rec_common * 100:.2f}%")
    else:
        print("  No common labels found.")
    
    print(f"\nExclusive Labels:")
    if acc_exclusive is not None:
        print(f"  Accuracy: {acc_exclusive:.2f}%")
        print(f"  F1-Score: {f1_exclusive * 100:.2f}%")
        print(f"  Precision: {prec_exclusive * 100:.2f}%")
        print(f"  Recall: {rec_exclusive * 100:.2f}%")
    else:
        print("  No exclusive labels found.")
    
    print(f"\nExclusive Binary (Normal vs Attack):")
    if acc_binary is not None:
        print(f"  Accuracy: {acc_binary:.2f}%")
        print(f"  F1-Score: {f1_binary * 100:.2f}%")
        print(f"  Precision: {prec_binary * 100:.2f}%")
        print(f"  Recall: {rec_binary * 100:.2f}%")
    else:
        print("  No exclusive labels found for binary classification.")

    return {
        'all': {'acc': acc_all, 'f1': f1_all * 100, 'prec': prec_all * 100, 'rec': rec_all * 100},
        'common': {'acc': acc_common, 'f1': f1_common * 100 if f1_common else None, 'prec': prec_common * 100 if prec_common else None, 'rec': rec_common * 100 if rec_common else None},
        'exclusive': {'acc': acc_exclusive, 'f1': f1_exclusive * 100 if f1_exclusive else None, 'prec': prec_exclusive * 100 if prec_exclusive else None, 'rec': rec_exclusive * 100 if rec_exclusive else None},
        'binary': {'acc': acc_binary, 'f1': f1_binary * 100 if f1_binary else None, 'prec': prec_binary * 100 if prec_binary else None, 'rec': rec_binary * 100 if rec_binary else None}
    }

# Main execution
if __name__ == "__main__":
    # Paths
    model_dir = f"models_KDD_{isModifDoss}"
    results_dir = f"results_KDD_{isModifDoss}"
    formated_test_path = f"formated_test_adv_{isModifDataset}.data"
    formated_train_path = f"formated_train_adv_{isModifDataset}.data"

    # Create results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load original model
    model_json_path = os.path.join(model_dir, f"defender_agent_model_.json")
    model_h5_path = os.path.join(model_dir, f"defender_agent_model_.h5")
    
    if not os.path.exists(model_json_path) or not os.path.exists(model_h5_path):
        raise FileNotFoundError(f"Original model files not found in {model_dir}: {model_json_path}, {model_h5_path}")

    with open(model_json_path, "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights(model_h5_path)
    model.compile(loss=huber_loss, optimizer="Adam")

    # Load test and train data
    env_test = RLenv('test', formated_test_path=formated_test_path)
    env_train = RLenv('train', formated_train_path=formated_train_path)
    test_states, test_labels = env_test.get_full()
    train_states, train_labels = env_train.get_full()

    # Debug: Print shapes and columns
    print(f"Test states shape: {test_states.shape}")
    print(f"Test states columns: {list(test_states.columns)}")
    print(f"Train states shape: {train_states.shape}")
    print(f"Train states columns: {list(train_states.columns)}")

    # Align test_states with train_states columns (101 features)
    expected_columns = train_states.columns
    if len(test_states.columns) != len(expected_columns):
        print(f"Warning: Test states has {len(test_states.columns)} features, expected {len(expected_columns)}")
        common_columns = [col for col in test_states.columns if col in expected_columns]
        if len(common_columns) < len(expected_columns):
            raise ValueError(f"Test data missing required features. Common columns: {len(common_columns)}, expected: {len(expected_columns)}")
        test_states = test_states[expected_columns]
        print(f"Adjusted test states to {len(test_states.columns)} features: {list(test_states.columns)}")

    # Map test labels to main categories and get detailed labels
    test_mapped_labels = []
    test_detailed_labels = []
    for _, label in test_labels.iterrows():
        detailed_label = label.idxmax()
        main_category = env_test.attack_map[detailed_label]
        test_mapped_labels.append(env_test.attack_types.index(main_category))
        test_detailed_labels.append(detailed_label)
    test_mapped_labels = np.array(test_mapped_labels)
    test_detailed_labels = np.array(test_detailed_labels)

    # Identify common and exclusive labels
    train_attack_names = set(env_train.attack_names)
    test_attack_names = set(env_test.attack_names)
    common_labels = train_attack_names.intersection(test_attack_names)
    exclusive_labels = test_attack_names - train_attack_names

    # Evaluate original model
    print("\nEvaluating Original Model")
    original_metrics = evaluate_model(model, test_states, test_labels, test_detailed_labels, env_test, results_dir, isModifDoss, "Original")

    # Adversarial Training
    print("\nStarting Adversarial Training")
    # Prepare training data
    train_mapped_labels = []
    for _, label in train_labels.iterrows():
        detailed_label = label.idxmax()
        main_category = env_train.attack_map[detailed_label]
        train_mapped_labels.append(env_train.attack_types.index(main_category))
    train_mapped_labels = np.array(train_mapped_labels)

    # Generate adversarial examples
    adversarial_train_states = generate_adversarial_examples(model, train_states.values, tf.one_hot(train_mapped_labels, depth=len(env_train.attack_types)), epsilon=0.1)
    adversarial_train_states = pd.DataFrame(adversarial_train_states, columns=train_states.columns)

    # Combine original and adversarial data
    combined_train_states = pd.concat([train_states, adversarial_train_states], ignore_index=True)
    combined_train_labels = np.concatenate([train_mapped_labels, train_mapped_labels])

    # Fine-tune the model
    model.fit(
        combined_train_states.values,
        tf.one_hot(combined_train_labels, depth=len(env_train.attack_types)),
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # Save fine-tuned model
    fine_tuned_suffix = f"{isModifDoss}_adversarial"
    fine_tuned_model_dir = f"models_KDD_{fine_tuned_suffix}"
    if not os.path.exists(fine_tuned_model_dir):
        os.makedirs(fine_tuned_model_dir)
    fine_tuned_json_path = os.path.join(fine_tuned_model_dir, f"defender_agent_model.json")
    fine_tuned_h5_path = os.path.join(fine_tuned_model_dir, f"defender_agent_model.h5")
    with open(fine_tuned_json_path, 'w') as jfile:
        jfile.write(model.to_json())
    model.save_weights(fine_tuned_h5_path)
    print(f"Fine-tuned model saved to: {fine_tuned_json_path}, {fine_tuned_h5_path}")

    # Evaluate fine-tuned model
    print("\nEvaluating Fine-Tuned Model")
    fine_tuned_metrics = evaluate_model(model, test_states, test_labels, test_detailed_labels, env_test, results_dir, fine_tuned_suffix, "Fine-Tuned")