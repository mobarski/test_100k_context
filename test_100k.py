import os

MODEL = "claude-instant-v1-100k"
OUTPUT_PATH = 'test_100k.tsv' # results will be appended
N_ITEMS = 430 # 20500->96k 10700->50k 5400->25k 430->2k
N_ITERS = 400

API_KEY = os.getenv('ANTHROPIC_API_KEY')

###############################################################################

from random import shuffle, randint
from time import time
from collections import Counter
from pprint import pprint
from tqdm import tqdm
from statistics import mean

import anthropic


def generate(N):
	"generate test data"
	data = set()
	while len(data) < N:
		x = randint(1,1_000_000_000)
		data.add(x)
	data = list(data)
	shuffle(data)
	data = [f"{x:x}" for x in data]
	return data


client = anthropic.Client(API_KEY)

def complete(prompt, system='', max_new_tokens=100, stop=[], temperature=0.0, model=MODEL):
	resp = client.completion(
		prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
		stop_sequences=[anthropic.HUMAN_PROMPT] + stop,
		model=model,
		max_tokens_to_sample=max_new_tokens,
		temperature=temperature,
	)
	return resp['completion']


def score(N, n=1, BUCKETS=5, verbose=False):
	"scoring loop"
	ok_cnt = Counter()
	all_cnt = Counter()
	tokens_cnt = []
	
	#
	for iter in tqdm(range(n)) if verbose else range(n):
		data = generate(N)
		data_str = ' '.join(data)
		n_tokens = anthropic.count_tokens(data_str)
		if verbose:
			print(f' {iter=} {n_tokens=}')

		selected_index = randint(0,len(data)-2)
		selected = data[selected_index]
		target = data[selected_index+1]

		# 1) task-question-data-answer
		prompt1 = f"""Analyze the following list of hexadecimal numbers and answer the question.
		Question: What's the next list element after {selected}?
		Data: {data_str}
		Answer:"""

		# 2) task-data-question-answer
		prompt2 = f"""Analyze the following list of hexadecimal numbers and answer the question.
		Data: {data_str}
		Question: What's the next list element after {selected}?
		Answer:"""

		# 3) data-question-answer
		prompt3 = f"""Data: {data_str}
		Question: What's the next list element after {selected}?
		Answer:"""

		# 4) question-data-answer
		prompt4 = f"""Question: What's the next list element after {selected}?
		Data: {data_str}
		Answer:"""
		
		bucket = BUCKETS*selected_index // N
		all_cnt[bucket] += 1
		#
		for p,prompt in zip(['1','2','3','4'],[prompt1,prompt2,prompt3,prompt4]):
			key = f'{p}:{bucket}'
			resp = complete(prompt)
			ok = target in resp
			ok_cnt[key] += int(ok)
			tokens_cnt += [n_tokens]
	
	return ok_cnt, all_cnt, tokens_cnt

from multiprocessing.pool import ThreadPool
def score_para(N,n,workers=4,BUCKETS=5):
	"parallel scoring"
	def worker(x):
		return score(N,1,BUCKETS=BUCKETS)
	ok_cnt = Counter()
	all_cnt = Counter()
	n_tok = []
	with ThreadPool(workers) as pool:
		out = pool.imap(worker, range(n))
		for ok, all, tok in tqdm(out):
			ok_cnt.update(ok)
			all_cnt.update(all)
			n_tok.extend(tok)
	return ok_cnt, all_cnt, n_tok

###############################################################################

if __name__=="__main__":
	N = N_ITEMS
	n = N_ITERS
	ok,all,n_tok = score_para(N, n, BUCKETS=5)
	out = []
	for k in ok:
		b = int(k.split(':')[1])
		p = k.split(':')[0]
		n_ok = ok[k]
		n_all = all[b]
		acc = n_ok / n_all
		out += [(acc, n_ok, n_all, p, b)]
	out.sort(reverse=True)
	with open(OUTPUT_PATH,'a') as fo:
		if fo.tell()==0:
			print('run_id n_items n_iters n_tok part prompt n_ok n_all acc'.replace(' ','\t'), file=fo)
		id = int(time())
		for acc,n_ok,n_all,p,b in out:
			print(f"part:{b+1} prompt:{p:4} {n_ok:3} {n_all:3} {acc:6.1%}")
			print(id, N, n, int(mean(n_tok)), f"part:{b+1}", f"prompt:{p}", n_ok, n_all, round(100*acc,1), file=fo, sep='\t')
	print(f'{N=} {n=} n_tok_avg={mean(n_tok):.0f}')
