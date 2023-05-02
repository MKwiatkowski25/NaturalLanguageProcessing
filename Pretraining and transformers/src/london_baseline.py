# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
def evaluate_london(filepath):
    with open(filepath) as fin:
        lines = [x.strip().split('\t') for x in fin]
        if len(lines[0]) == 1:
            print('No gold birth places provided; returning (0,0)')
            return None
        true_places = [x[1] for x in lines]
        total = len(true_places)
        correct = len(list(filter(lambda x: x[0] == x[1],
                                  zip(true_places, total*['London']))))
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('No targets provided')