package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;
import java.util.HashSet;
import java.util.Set;
/*
 * Word alignment model that uses IBM model 1.
 * @author Veni Johanna
 * @author Kat Busch
 */

public class M1WordAligner implements WordAligner {
  private CounterMap<String, String> sourceTargetProbability;


  public Alignment align(SentencePair sentencePair) {
    Alignment alignment = new Alignment();
   
    List<String> targetWords = sentencePair.getTargetWords();
    List<String> sourceWords = sentencePair.getSourceWords();

    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();

    for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
      String sourceWord = sourceWords.get(srcIndex);

      int bestWordIndex = 0;
      double bestProbability = 0.0;

      for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
        String targetWord = targetWords.get(tgtIndex);

        if (sourceTargetProbability.getCount(sourceWord, targetWord) >
            bestProbability) {
          bestProbability = sourceTargetProbability.getCount(sourceWord,
                                                             targetWord);
          bestWordIndex = tgtIndex;

        }
      }
      alignment.addPredictedAlignment(bestWordIndex, srcIndex);
    }
    return alignment;
  }

  private void printDebug(List<SentencePair> trainingPairs) {
    for (SentencePair pair : trainingPairs) {
              
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();

      for (String targetWord : targetWords) {
        for (String sourceWord : sourceWords) {
          System.out.println(sourceWord + " " + targetWord + " " + sourceTargetProbability.getCount(sourceWord, targetWord));
        }
      }
    }
  }
  public void train(List<SentencePair> trainingPairs) {
    Set<String> allTargetWords = new HashSet<String>();
    Set<String> allSourceWords = new HashSet<String>();

    for (SentencePair pair : trainingPairs) {
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();
      for (int i = 0; i < targetWords.size(); i++) {
        targetWords.set(i, targetWords.get(i).toLowerCase());
      }

      for (int i = 0; i < sourceWords.size(); i++) {
        sourceWords.set(i, sourceWords.get(i).toLowerCase());
      }

      allTargetWords.addAll(targetWords);
      allSourceWords.addAll(sourceWords);    

      sourceWords.add(0, NULL_WORD);
    }

    initUniformProbability(allSourceWords, allTargetWords);

    boolean converged = false;
    int numIter = 0;
    do {
      converged = trainOnce(trainingPairs, allSourceWords, allTargetWords);
  
      numIter++;
    }while ((!converged) && (numIter < 10));  

    for (SentencePair pair: trainingPairs) {
      List<String> sourceWords = pair.getSourceWords();
      sourceWords.remove(0);
    }
  }

  private boolean trainOnce(List<SentencePair> trainingPairs,
                         Set<String> allSourceWords, Set<String> allTargetWords) {
    CounterMap<String, String> sourceTargetCooccurrenceCount = new CounterMap<String, String>();

    for (SentencePair pair : trainingPairs) {
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();

      int numTargetWords = targetWords.size();
      int numSourceWords = sourceWords.size();

      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        double denom = 0.0;

        for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
          denom += sourceTargetProbability.getCount(sourceWords.get(srcIndex),
                                                    targetWords.get(tgtIndex));
        }

        for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {        
          double posterior = sourceTargetProbability.getCount(sourceWords.get(srcIndex),
                                                       targetWords.get(tgtIndex)) / (double) denom;
          sourceTargetCooccurrenceCount.incrementCount(sourceWords.get(srcIndex),
                                                      targetWords.get(tgtIndex),
                                                      posterior);
        }
      }
    }

    int numConverged = 0;
    // Normalizing the counts into probability
    for (String targetWord : allTargetWords) {
      double denom = 0;
      for (String sourceWord : allSourceWords) {
        denom += sourceTargetCooccurrenceCount.getCount(sourceWord, targetWord);
      }

      for (String sourceWord : allSourceWords) {
        double newValue = sourceTargetCooccurrenceCount.getCount(
                                                sourceWord, targetWord) / denom;

        if (Math.abs(sourceTargetProbability.getCount(sourceWord, targetWord) -
                     newValue) < 0.000001) {
            numConverged++;
    
        }
        sourceTargetProbability.setCount(sourceWord, targetWord, newValue);

      }
    }

    return (numConverged > 0.95 * allSourceWords.size() * allTargetWords.size());
  }
  // Initializes sourceTargetProbability counterMap with uniform probability.

  private void initUniformProbability(Set<String> allSourceWords, Set<String> allTargetWords) {
    sourceTargetProbability = new CounterMap<String, String>();

    for (String sourceWord : allSourceWords) {
      for (String targetWord : allTargetWords) {
        sourceTargetProbability.setCount(sourceWord, targetWord, 1/(double)allSourceWords.size());
      }
    }
  }

}
