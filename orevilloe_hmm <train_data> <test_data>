import numpy as np

def load_data(file_path):
    """
    Load sequences and their state annotations from a specified file.
    
    Parameters:
    - file_path: Path to the file containing the data.
    
    Returns:
    - A tuple containing two lists: one for sequences and one for their corresponding state annotations.
    """
    with open(file_path, 'r') as file:
        raw_data = file.read().strip().split('\n\n')
    seq_list = []
    state_list = []
    for entry in raw_data:
        parts = entry.split('\n')
        if len(parts) == 3:
            seq_list.append(parts[1].strip())
            state_list.append(parts[2].strip().lstrip('#').strip())
    return seq_list, state_list

def tally_trans_emissions(seq_list, state_list):
    """
    Count transitions between states and the emission of symbols from states.
    
    Parameters:
    - seq_list: A list of sequences.
    - state_list: A list of state sequences corresponding to the sequences.
    
    Returns:
    - A tuple of two dictionaries: the first for transition counts and the second for emission counts.
    """
    valid_states = {'o', 'M', 'i'}
    trans_count = {s: {s2: 0 for s2 in valid_states} for s in valid_states}
    emit_count = {s: {} for s in valid_states}
    
    for seq, state_seq in zip(seq_list, state_list):
        prev_s = None
        for i, current_s in enumerate(state_seq):
            if current_s not in valid_states:
                print(f"Unexpected state '{current_s}' encountered. Skipping.")
                continue

            symbol = seq[i]
            if symbol not in emit_count[current_s]:
                emit_count[current_s][symbol] = 1
            else:
                emit_count[current_s][symbol] += 1
            
            if prev_s is not None:
                trans_count[prev_s][current_s] += 1
            prev_s = current_s
    
    return trans_count, emit_count

def compute_probs(trans_count, emit_count):
    """
    Calculate transition and emission probabilities from counts, applying smoothing.
    
    Parameters:
    - trans_count: Dictionary of transition counts.
    - emit_count: Dictionary of emission counts.
    
    Returns:
    - A tuple of two dictionaries: transition probabilities and emission probabilities.
    """
    smooth_val = 1e-3
    states = ['o', 'M', 'i']
    symbols = set(sym for em in emit_count.values() for sym in em)

    trans_probs = {s: {} for s in trans_count}
    for s in states:
        total_trans = sum(trans_count.get(s, {}).values()) + smooth_val * len(states)
        for s2 in states:
            trans_probs[s][s2] = (trans_count.get(s, {}).get(s2, 0) + smooth_val) / total_trans

    emit_probs = {s: {} for s in emit_count}
    for s in states:
        total_emits = sum(emit_count.get(s, {}).values()) + smooth_val * len(symbols)
        for sym in symbols:
            emit_probs[s][sym] = (emit_count.get(s, {}).get(sym, 0) + smooth_val) / total_emits

    return trans_probs, emit_probs

def viterbi_algorithm(sequence, trans_probs, emit_probs):
    """
    Implement the Viterbi algorithm using log probabilities to find the most likely state sequence.
    
    Parameters:
    - sequence: The observed sequence for which to predict the state sequence.
    - trans_probs: Transition probabilities.
    - emit_probs: Emission probabilities.
    
    Returns:
    - The most likely state sequence for the given observation sequence.
    """
    state_idx = {'o': 0, 'M': 1, 'i': 2}
    log_prob_matrix = np.full((3, len(sequence)), -1000)
    state_path_matrix = np.zeros((3, len(sequence)), dtype=int)
    state_keys = ['o', 'M', 'i']

    for state in state_keys:
        log_prob_matrix[state_idx[state], 0] = np.log(emit_probs[state].get(sequence[0], 1e-10))

    for i in range(1, len(sequence)):
        for current_state in state_keys:
            for prev_state in state_keys:
                temp_prob = log_prob_matrix[state_idx[prev_state], i-1] + np.log(trans_probs[prev_state].get(current_state, 1e-10)) + np.log(emit_probs[current_state].get(sequence[i], 1e-10))
                if temp_prob > log_prob_matrix[state_idx[current_state], i]:
                    log_prob_matrix[state_idx[current_state], i] = temp_prob
                    state_path_matrix[state_idx[current_state], i] = state_idx[prev_state]

    final_state = np.argmax(log_prob_matrix[:, -1])
    optimal_path = [final_state]
    for i in range(len(sequence) - 1, 0, -1):
        optimal_path.insert(0, state_path_matrix[optimal_path[0], i])

    optimal_path_str = ''.join(state_keys[i] for i in optimal_path)
    return optimal_path_str

def load_test_data(test_file_path):
    """
    Load test sequences from a specified file, assuming no state annotations are present.
    
    Parameters:
    - test_file_path: Path to the test data file.
    
    Returns:
    - A list of test sequences.
    """
    with open(test_file_path, 'r') as file:
        raw_data = file.read().strip().split('\n\n')
    test_seq_list = []
    for entry in raw_data:
        parts = entry.split('\n')
        if len(parts) >= 2:
            test_seq_list.append(parts[1].strip())
    return test_seq_list

def evaluate_performance(true_states, predicted_states):
    """
    Evaluate the performance of the predicted state sequences against the true state sequences.
    
    Parameters:
    - true_states: A list of true state sequences.
    - predicted_states: A list of predicted state sequences.
    
    Prints the recall, precision, F1-score, and accuracy for each state label.
    """
    labels = ['M', 'i', 'o']
    metrics = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for label in labels}

    for true_seq, pred_seq in zip(true_states, predicted_states):
        for i in range(len(true_seq)):
            for label in labels:
                if pred_seq[i] == label:
                    if true_seq[i] == label:
                        metrics[label]['TP'] += 1
                    else:
                        metrics[label]['FP'] += 1
                else:
                    if true_seq[i] == label:
                        metrics[label]['FN'] += 1
                    else:
                        metrics[label]['TN'] += 1

    for label in labels:
        TP = metrics[label]['TP']
        FP = metrics[label]['FP']
        TN = metrics[label]['TN']
        FN = metrics[label]['FN']
        
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        F1 = 2 * (recall * precision) / (recall + precision) if recall + precision != 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
        
        print(f"Label {label}: Recall={recall:.4f}, Precision={precision:.4f}, F1-Score={F1:.4f}, Accuracy={accuracy:.4f}")

def main():
    """
    Main function orchestrating the data loading, model training, prediction, and performance evaluation.
    """
    # Load training data
    train_seq_list, train_state_list = load_data('hw2_train.data') # Update 'hw2_train.data' with actual path
    trans_count, emit_count = tally_trans_emissions(train_seq_list, train_state_list)
    trans_probs, emit_probs = compute_probs(trans_count, emit_count)
    
    # Load test data and true state sequences for evaluation
    test_seq_list, true_state_list = load_data('hw2_test.data') # Update 'hw2_train.data' with actual path
    
    # Predict state sequences for the test data
    predicted_state_list = []
    for seq in test_seq_list:
        pred_states = viterbi_algorithm(seq, trans_probs, emit_probs)
        predicted_state_list.append(pred_states)
    
    # Evaluate and print performance metrics
    evaluate_performance(true_state_list, predicted_state_list)

if __name__ == "__main__":
    main()
