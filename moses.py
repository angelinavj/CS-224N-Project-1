import sys
import os
import time
import commands
import subprocess

HOME = "/afs/ir/users/k/b/kbusch/cs224n/pa1-mt"
MOSES = "/afs/ir/class/cs224n/bin/mosesdecoder"
GIZA = "/afs/ir/class/cs224n/bin/giza-pp-read-only/external-bin-dir"

MERT = ""
PRO = "--pairwise-ranked"

def run_command(name, command, stdin=None, stdout=None):
	outfile.write('Running %s\n' %(name))
	outfile.write('Command: \n %s' % (command))
	outfile.flush()
	start = time.time()
	#output = commands.getoutput(command)
	if stdin or stdout:
		subprocess.call(command, stdin=stdin, stdout=stdout)
	else:
		os.system(command)
	end = time.time()
	outfile.write('\nRuntime: %d\n\n'%(end - start))
	outfile.flush()

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
	command = ['%s/bin/moses'%(MOSES), '-du',
		'-f', '%s/tune/moses.ini'%(HOME)]
	run_command("decoding", command, 
		stdin=open('%s/mt-dev-test.fr'%(HOME)),
		stdout=open('%s/output/%s-test.out'%(HOME, name), 'w'))

	os.system("cp %s/tune/moses.ini %s/output/%s-moses.ini"%(HOME, HOME, name))

def eval(name):
	command = ['%s/scripts/generic/multi-bleu.perl'%(MOSES),
	 	'%s/mt-dev-test.en'%(HOME)]

	output = run_command("evaluation", command, 
		stdin=open('%s/output/%s-test.out'%(HOME, name)),
		stdout=open('%s/output/%s-eval.out'%(HOME, name), 'w'))
	#with open('%s/output/%s-eval.out'%(HOME, name), 'w') as output_file:
	#	output_file.write(output)

def main(phrase_len, dist_lim):
	#gen_phrase_table(phrase_len)
	name = 's50000-%dpl-%ddl'%(phrase_len, dist_lim)	
	#outfile.write('\n\nMERT:\n')
	#tune(dist_lim, MERT)
	#decode(name + '-mert')
	#eval(name + '-mert')

	outfile.write('\n\nPRO:\n')
	tune(dist_lim, PRO)
	decode(name + '-pro')
	eval(name + '-pro')

if __name__ == '__main__':
	phrase_len = 6
	for dist_lim in [4]:
		outfile = open('moses_output_50k_%s_%s.out'%(phrase_len, dist_lim), 'w')
		main(int(phrase_len), int(dist_lim))
		outfile.close()
