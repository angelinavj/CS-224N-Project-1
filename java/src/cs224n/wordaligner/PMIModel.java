package cs224n.wordaligner;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import cs224n.util.Counter;
import cs224n.util.CounterMap;

/**
 *  Uses simple PMI to computer alignments
 *  @author Kat Busch
 */

public class PMIModel implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;

	private CounterMap<String,String> pairwiseCounts;
	private Counter<String> srcLangCounts;
	private Counter<String> targetLangCounts;
	
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment a = new Alignment();
		List<String> sourceWords = sentencePair.getSourceWords();
		List<String> targetWords = sentencePair.getTargetWords();
		for(int i = 0; i < sourceWords.size(); i++) {
			int max_j = -1;
			double max_prob = 0;
			String srcWord = sourceWords.get(i);
			for(int j = 0; j < targetWords.size(); j++) {
				String targetWord = targetWords.get(j);
				double prob = pairwiseCounts.getCount(srcWord, targetWord);
				prob /= targetLangCounts.getCount(targetWord);
				if(prob > max_prob) {
					max_prob = prob;
					max_j = j;
				}
			}
			a.addPredictedAlignment(max_j, i);
		}
		return a;
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		pairwiseCounts = new CounterMap<String, String>();
		srcLangCounts = new Counter<String>();
		targetLangCounts = new Counter<String>();
		for(SentencePair p: trainingData) {
			Set<String> sourceWords = new HashSet<String>(p.getSourceWords());
			srcLangCounts.incrementAll(sourceWords, 1);
			Set<String> targetWords = new HashSet<String>(p.getTargetWords());
			targetLangCounts.incrementAll(targetWords, 1);
			for(String srcWord: sourceWords) {
				for(String targetWord: targetWords){
					pairwiseCounts.incrementCount(srcWord, targetWord, 1);
				}
			}
		}
	}
}
