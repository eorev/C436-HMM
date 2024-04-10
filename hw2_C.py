import numpy as np

def read_data(filename):
    with open(filename, 'r') as file:
        data = file.read().strip().split('\n\n')
    sequences = []
    state_sequences = []
    for entry in data:
        parts = entry.split('\n')
        if len(parts) == 3:  # Ensure we have both a sequence and its state sequence
            sequences.append(parts[1].strip())
            # Remove the initial '#' from the state sequence and any trailing whitespace
            state_sequences.append(parts[2].strip().lstrip('#').strip())
    return sequences, state_sequences

def count_transitions_and_emissions(sequences, state_sequences):
    # Define the expected states
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
    transition_probs = {s: {} for s in transitions}
    for s, targets in transitions.items():
        total = sum(targets.values())
        for t, count in targets.items():
            transition_probs[s][t] = count / total if total > 0 else 0
    
    emission_probs = {s: {} for s in emissions}
    for s, aas in emissions.items():
        total = sum(aas.values())
        for aa, count in aas.items():
            emission_probs[s][aa] = count / total if total > 0 else 0
    
    return transition_probs, emission_probs

def log_viterbi(sequences, transition_probs, emission_probs):
    state_map = {'o': 0, 'M': 1, 'i': 2}  # Map states to indices for easier access
    state_list = ['o', 'M', 'i']  # List of states to maintain order
    
    for seq in sequences:
        # Initialization
        V = np.zeros((len(state_list), len(seq)))  # Viterbi matrix
        path = np.zeros((len(state_list), len(seq)), dtype=int)  # Path matrix
        for s in state_list:
            V[state_map[s], 0] = np.log(emission_probs[s].get(seq[0], 1e-10))  # Use a small value if emission prob is 0
        
        # Recursion
        for t in range(1, len(seq)):
            for s in state_list:
                prob_list = V[:, t-1] + [np.log(transition_probs[ps].get(s, 1e-10)) for ps in state_list] + np.log(emission_probs[s].get(seq[t], 1e-10))
                V[state_map[s], t] = np.max(prob_list)
                path[state_map[s], t] = np.argmax(prob_list)
        
        # Termination
        best_path_prob = np.max(V[:, -1])
        best_last_state = np.argmax(V[:, -1])
        
        # Traceback
        best_path = [best_last_state]
        for t in range(len(seq) - 1, 0, -1):
            best_path.insert(0, path[best_path[0], t])
        
        best_path = [state_list[i] for i in best_path]
        
        print("Best path:", ''.join(best_path))
        print("Best path probability (log):", best_path_prob)

def read_test_data(filename):
    """
    This function reads sequences from a test data file where each sequence
    is expected to be in its own section, without state sequences.
    """
    with open(filename, 'r') as file:
        data = file.read().strip().split('\n\n')
    sequences = []
    for entry in data:
        parts = entry.split('\n')
        if len(parts) >= 2:  # Checking for at least a header and sequence
            sequences.append(parts[1].strip())
    return sequences

# Main function to run the analysis
def main():
    sequences, state_sequences = read_data('hw2_train.data')
    transitions, emissions = count_transitions_and_emissions(sequences, state_sequences)
    transition_probs, emission_probs = calculate_probabilities(transitions, emissions)
    
    # Load test data
    test_sequences = read_test_data('hw2_test.data')
    
    # Predict states for each test sequence using the Viterbi algorithm
    for seq in test_sequences:
        log_viterbi([seq], transition_probs, emission_probs)

if __name__ == "__main__":
    main()
