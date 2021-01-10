import math
import pandas as pd

def binary_adaboost_fit_weights(df, target=None):
	amount_of_say_dict = {}
	if target is None:
		target = df.iloc[:, -1]
	else:
		target = df[target].tolist()
	for col_name, values in df.iteritems():
		results = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
		for i, v in enumerate(values):
			if v == 1 and target[i] == 1:
				results['tp'] += 1
			elif v == 1 and target[i] == 0:
				results['fp'] += 1
			elif v == 0 and target[i] == 1:
				results['fn'] += 1
			else:
				results['tn'] += 1
		total_error = results['fp'] + results['fn'] / len(target)
		amount_of_say = 0.5 * math.log((1 - total_error) / total_error, math.e)
		amount_of_say_dict[col_name] = amount_of_say

	return amount_of_say_dict