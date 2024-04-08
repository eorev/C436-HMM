import numpy as np

def read_data(filename):
    """
    This function reads the training data from a file.
    It expects the data to be in a specific format:
    - Each entry is separated by two newline characters.
    - Each entry consists of three lines: a header, a sequence, and a state sequence.
    The function returns two lists: sequences and their corresponding state sequences.
    """
    with open(filename, 'r') as file:
        data = file.read().strip().split('\n\n')
    sequences = []
    state_sequences = []
    for entry in data:
        parts = entry.split('\n')
        if len(parts) == 3:
            sequences.append(parts[1].strip())
            state_sequences.append(parts[2].strip().lstrip('#').strip())
    return sequences, state_sequences

def count_transitions_and_emissions(sequences, state_sequences):
    """
    This function counts the transitions between states and emissions of symbols for each state.
    It takes the sequences and their corresponding state sequences as input.
    It returns two dictionaries: transitions and emissions.
    - transitions: counts of transitions between states.
    - emissions: counts of symbols emitted from each state.
    """
    states = {'o', 'M', 'i'}
    transitions = {s: {s2: 0 for s2 in states} for s in states}
    emissions = {s: {} for s in states}
    
    for seq, states_seq in zip(sequences, state_sequences):
        prev_state = None
        for i, state in enumerate(states_seq):
            if state not in states:
                print(f"Unexpected state '{state}' found. Skipping.")
                continue

            aa = seq[i]  # amino acid
            if aa not in emissions[state]:
                emissions[state][aa] = 1
            else:
                emissions[state][aa] += 1
            
            if prev_state is not None:
                transitions[prev_state][state] += 1
            prev_state = state
    
    return transitions, emissions

def calculate_probabilities(transitions, emissions):
    """
    This function calculates the transition and emission probabilities based on the counts.
    It applies smoothing to handle unseen transitions and emissions.
    It returns two dictionaries: transition_probs and emission_probs.
    - transition_probs: probabilities of transitioning from one state to another.
    - emission_probs: probabilities of emitting a symbol from each state.
    """
    smoothing_val = 1e-3
    all_states = ['o', 'M', 'i']
    all_symbols = set(sym for em in emissions.values() for sym in em)

    transition_probs = {s: {} for s in transitions}
    for s in all_states:
        total_transitions = sum(transitions.get(s, {}).values()) + smoothing_val * len(all_states)
        for s2 in all_states:
            transition_probs[s][s2] = (transitions.get(s, {}).get(s2, 0) + smoothing_val) / total_transitions

    emission_probs = {s: {} for s in emissions}
    for s in all_states:
        total_emissions = sum(emissions.get(s, {}).values()) + smoothing_val * len(all_symbols)
        for symbol in all_symbols:
            emission_probs[s][symbol] = (emissions.get(s, {}).get(symbol, 0) + smoothing_val) / total_emissions

    return transition_probs, emission_probs

def log_viterbi(seq, transition_probs, emission_probs):
    """
    This function implements the Viterbi algorithm with logarithm transformation.
    It takes a sequence, transition probabilities, and emission probabilities as input.
    It returns the most likely state sequence for the given sequence.
    """
    state_map = {'o': 0, 'M': 1, 'i': 2}
    V = np.full((3, len(seq)), -1000)
    path = np.zeros((3, len(seq)), dtype=int)
    state_list = ['o', 'M', 'i']

    for s in state_list:
        V[state_map[s], 0] = np.log(emission_probs[s].get(seq[0], 1e-10))

    for t in range(1, len(seq)):
        for s in state_list:
            for ps in state_list:
                prob = V[state_map[ps], t-1] + np.log(transition_probs[ps].get(s, 1e-10)) + np.log(emission_probs[s].get(seq[t], 1e-10))
                if prob > V[state_map[s], t]:
                    V[state_map[s], t] = prob
                    path[state_map[s], t] = state_map[ps]

    best_last_state = np.argmax(V[:, -1])
    best_path = [best_last_state]
    for t in range(len(seq) - 1, 0, -1):
        best_path.insert(0, path[best_path[0], t])

    best_path_str = ''.join(state_list[i] for i in best_path)
    return best_path_str

def read_test_data(filename):
    """
    This function reads sequences from a test data file.
    It expects each sequence to be in its own section, without state sequences.
    It returns a list of sequences.
    """
    with open(filename, 'r') as file:
        data = file.read().strip().split('\n\n')
    sequences = []
    for entry in data:
        parts = entry.split('\n')
        if len(parts) >= 2:
            sequences.append(parts[1].strip())
    return sequences

def evaluate_performance(true_state_sequences, predicted_state_sequences):
    """
    This function evaluates the performance of the predicted state sequences.
    It compares the predicted state sequences with the true state sequences.
    It calculates and prints the recall, precision, F1-score, and accuracy for each label.
    """
    labels = ['M', 'i', 'o']
    metrics = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for label in labels}

    for true_states, predicted_states in zip(true_state_sequences, predicted_state_sequences):
        true_states = true_states.upper()
        predicted_states = predicted_states.upper()
        
        for i in range(len(true_states)):
            for label in labels:
                if predicted_states[i] == label:
                    if true_states[i] == label:
                        metrics[label]['TP'] += 1
                    else:
                        metrics[label]['FP'] += 1
                else:
                    if true_states[i] == label:
                        metrics[label]['FN'] += 1
                    else:
                        metrics[label]['TN'] += 1

    for label in labels:
        TP = metrics[label]['TP']
        FP = metrics[label]['FP']
        TN = metrics[label]['TN']
        FN = metrics[label]['FN']
        
        R = TP / (TP + FN) if TP + FN != 0 else 0
        P = TP / (TP + FP) if TP + FP != 0 else 0
        F1 = 2 * (R * P) / (R + P) if R + P != 0 else 0
        ACCU = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
        
        print(f"Label {label}: Recall (R)={R:.4f}, Precision (P)={P:.4f}, F1-Score={F1:.4f}, Accuracy (ACCU)={ACCU:.4f}")

def main():
    """
    The main function that orchestrates the entire analysis.
    It loads the training data, calculates model parameters, loads the test data,
    predicts the state sequences for the test sequences, and evaluates the performance.
    """
    train_sequences, train_state_sequences = read_data('hw2_train.data')
    transitions, emissions = count_transitions_and_emissions(train_sequences, train_state_sequences)
    transition_probs, emission_probs = calculate_probabilities(transitions, emissions)
    
    test_sequences, true_state_sequences = read_data('hw2_test.data')
    
    predicted_state_sequences = []
    for seq in test_sequences:
        predicted_states = log_viterbi(seq, transition_probs, emission_probs)
        predicted_state_sequences.append(predicted_states)
    
    evaluate_performance(true_state_sequences, predicted_state_sequences)


if __name__ == "__main__":
    main()