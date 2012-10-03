package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
/*
 * Word alignment model that uses IBM model 1.
 * @author Veni Johanna
 * @author Kat Busch
 */

public class Model1 implements WordAligner {
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
    Set<String> allSourceWords = new HashSet<String>();

    // Set of target words that appears with a particular source word.
    HashMap coOccurringTargetWords = new HashMap();


    for (SentencePair pair : trainingPairs) {
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();
      for (int i = 0; i < targetWords.size(); i++) {
        targetWords.set(i, targetWords.get(i).toLowerCase());
      }

      sourceWords.add(0, NULL_WORD);

      for (int i = 0; i < sourceWords.size(); i++) {
        sourceWords.set(i, sourceWords.get(i).toLowerCase());
        for (String targetWord : targetWords) {
          if (coOccurringTargetWords.get(sourceWords.get(i)) == null) {
            HashSet<String> set = new HashSet<String>();
            set.add(targetWord);
            coOccurringTargetWords.put(sourceWords.get(i), set);
          } else {
            HashSet set = (HashSet) coOccurringTargetWords.get(sourceWords.get(i));
            set.add(targetWord);
            coOccurringTargetWords.put(sourceWords.get(i), set);
          }

        }
      }

      allSourceWords.addAll(sourceWords);    
    }

    initUniformProbability(allSourceWords, coOccurringTargetWords);

    boolean converged = false;
    int numIter = 0;
    do {
      converged = trainOnce(trainingPairs, allSourceWords, coOccurringTargetWords);
  
      numIter++;
    }while (!converged); 

    for (SentencePair pair: trainingPairs) {
      List<String> sourceWords = pair.getSourceWords();
      sourceWords.remove(0);
    }
  }

  public CounterMap<String, String> getTranslationProbability(List<SentencePair> trainingPairs) {
    train(trainingPairs);
    return sourceTargetProbability;
  }
  /*
   * Train the trainingPairs onc.
   * Returns true if the data converged in this training iteration, or false otherwise.
   */
  private boolean trainOnce(List<SentencePair> trainingPairs,
                         Set<String> allSourceWords, HashMap coOccurringTargetWords) {
    CounterMap<String, String> sourceTargetCooccurrenceCount = new CounterMap<String, String>();
    Counter<String> total = new Counter<String>();

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
          total.incrementCount(targetWords.get(tgtIndex), posterior);
        }
      }
    }

    int numConverged = 0;
    int totalData = 0;
    for (String sourceWord : allSourceWords) {
      for (String targetWord : (HashSet<String>)(coOccurringTargetWords.get(sourceWord))) {

        double newValue = sourceTargetCooccurrenceCount.getCount(
                                                sourceWord, targetWord) / total.getCount(targetWord);

        if (Math.abs(sourceTargetProbability.getCount(sourceWord, targetWord) -
                     newValue) < 0.000001) {
            numConverged++;
    
        }
        totalData++;
        sourceTargetProbability.setCount(sourceWord, targetWord, newValue);

      }
    }

    return (numConverged > 0.95 * totalData);
  }
  // Initializes sourceTargetProbability counterMap with uniform probability.

  private void initUniformProbability(Set<String> allSourceWords, HashMap coOccurringTargetWords) {
    sourceTargetProbability = new CounterMap<String, String>();

    for (String sourceWord : allSourceWords) {
      HashSet<String> targetWords = (HashSet<String>) (coOccurringTargetWords.get(sourceWord));
      for (String targetWord : targetWords) {
        double denom = targetWords.size();
        sourceTargetProbability.setCount(sourceWord, targetWord, 1/denom);
      }
    }
  }

}
