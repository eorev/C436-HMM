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

# Main function to run the analysis
def main():
    sequences, state_sequences = read_data('hw2_train.data')
    transitions, emissions = count_transitions_and_emissions(sequences, state_sequences)
    transition_probs, emission_probs = calculate_probabilities(transitions, emissions)
    
    print("Transition Probabilities:")
    for s, targets in transition_probs.items():
        for t, prob in targets.items():
            print(f"{s} -> {t}: {prob}")
    print("\nEmission Probabilities:")
    for s, aas in emission_probs.items():
        for aa, prob in aas.items():
            print(f"{s} emits {aa}: {prob}")

if __name__ == "__main__":
    main()
