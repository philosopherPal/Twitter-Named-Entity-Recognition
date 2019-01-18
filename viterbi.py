import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    dp = np.zeros((N, L))
    dp = np.full((N, L), -np.inf)
    max_last_edge_wt = -np.inf
    back_ptr = np.zeros((N, L))
    back_ptr_end = 0
    for i in range(0, L):
        dp[0][i] = emission_scores[0, i] + start_scores[i]
        back_ptr[0, i] = 0
        # print dp[0][i]
    for i in range(1, N):
        for j in range(0, L):
            for k in range(0, L):
                temp = emission_scores[i, j] + dp[i - 1, k] + trans_scores[k, j]
                # print("temp, i , j, k:", temp, i, j, k)
                if temp > dp[i, j]:
                    dp[i, j] = temp
                    back_ptr[i, j] = k

    for i in range(0, L):
        temp = dp[N - 1][i] + end_scores[i]
        if temp > max_last_edge_wt:
            max_last_edge_wt = temp
            back_ptr_end = i
    # print(back_ptr)
    # print(back_ptr_end)
    # print(start_scores)
    # print(end_scores)
    # print (trans_scores)
    # print(emission_scores)
    # print(end_scores)
    # print(dp)
    tagged_sequence = list()
    tagged_sequence.append(int(back_ptr_end))
    for i in range(N - 1, 0, -1):
        # print(i, back_ptr_end)
        tagged_sequence.append(int(back_ptr[i][back_ptr_end]))
        back_ptr_end = int(back_ptr[i][back_ptr_end])
    # exit()
    # y = []
    # for i in xrange(N):
    #     # stupid sequence
    #     y.append(i % L)
    # score set to 0
    # print(y)
    return (max_last_edge_wt, list(reversed(tagged_sequence)))
