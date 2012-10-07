import sys
import os
import time

HOME = "/afs/ir/users/k/b/kbusch/cs224n/pa1-mt"
MOSES = "/afs/ir/class/cs224n/bin/mosesdecoder"
GIZA = "/afs/ir/class/cs224n/bin/giza-pp-read-only/external-bin-dir"

MERT = ""
PRO = "--pairwise-ranked"

def run_command(name, command):
	outfile.write('Running %s\n' %(name))
	outfile.write('Command: \n %s' % (command))

	tmp = open("tempout", 'w')
	tmp.write(command)
	tmp.close()

	start = time.time()
	os.system(command)
	end = time.time()
	outfile.write('\nRuntime: %d'%(end - start))

def gen_phrase_table(phrase_len):
	command = ('%s/scripts/training/train-model.perl --max-phrase-length %d '
		'--external-bin-dir %s --first-step 4 --last-step 9 '
		'-root-dir %s/train -corpus %s/training/corpus -f f -e e '
		'-alignment-file %s/training/corpus -alignment align '
		'-lm 0:3:"%s"/lm.bin:8' % 
		(MOSES, phrase_len, GIZA, HOME, HOME, HOME, HOME) )
	run_command('generate phrase table', command)

def tune(dist_lim, alg):
	command = ('%s/scripts/training/mert-moses.pl '
		'--working-dir %s/tune %s '
		'--decoder-flags="-distortion-limit %d -threads 4" %s/mt-dev.fr '
		'%s/mt-dev.en %s/bin/moses %s/train/model/moses.ini '
		'--mertdir %s/bin/' %
		(MOSES, HOME, alg, dist_lim, HOME, HOME, MOSES, HOME, MOSES) )
	run_command("tuning", command) 

def decode(name):
	command = ('cat %s/mt-dev-test.fr | %s/bin/moses -du '
		'-f %s/tune/moses.ini > %s/output/%s-test.out' %
		(HOME, MOSES, HOME, HOME, name) )
	run_command("decoding", command)
	os.system("cp %s/tune/moses.ini %s/output/%s-moses.init"%(HOME, HOME, name))

def eval(name):
	command = ('%s/scripts/generic/multi-bleu.perl '
	 	'%s/mt-dev-test.en < %s/output/%s-test.out > %s/output/%s-eval.out' %
	 	(MOSES, HOME, HOME, name, HOME, name) )
	run_command("evaluation", command)

def main(phrase_len, dist_lim):
	#gen_phrase_table(phrase_len)
	name = 's2000-%dpl-%ddl'%(phrase_len, dist_lim)

	outfile.write('\n\nMERT:\n')
	tune(dist_lim, MERT)
	decode(name + '-mert')
	eval(name + '-mert')

	outfile.write('\n\nPRO:\n')
	tune(dist_lim, PRO)
	name = 's2000-%dpl-%ddl'%(phrase_len, dist_lim)
	decode(name + '-pro')
	eval(name + '-pro')

if __name__ == '__main__':
	outfile = open('moses_output_%s_%s.out'%(sys.argv[1], sys.argv[2]), 'w')
	main(int(sys.argv[1]), int(sys.argv[2]))
	outfile.close()