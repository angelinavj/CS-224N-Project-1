package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Random;
/*
 * Word alignment model that uses IBM model 1.
 * @author Veni Johanna
 * @author Kat Busch
 */

public class Model2 implements WordAligner {
  private CounterMap<String, String> sourceTargetProbability;
  private CounterMap<Integer, Integer> alignmentProbability;
  private int maxNumSourceWords;
  private int maxNumTargetWords;

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
        double probability = sourceTargetProbability.getCount(sourceWord, targetWord) *
                                alignmentProbability.getCount(srcIndex, tgtIndex);
        if (probability > bestProbability) {
          bestProbability = probability;
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

    initProbabilities(trainingPairs, allSourceWords, coOccurringTargetWords);

    boolean converged = false;
    int numIter = 0;
    do {
      converged = trainOnce(trainingPairs, allSourceWords, coOccurringTargetWords);
 
      numIter++;
    }while ((!converged) && (numIter < 50)); 

    for (SentencePair pair: trainingPairs) {
      List<String> sourceWords = pair.getSourceWords();
      sourceWords.remove(0);
    }
  }

  /*
   * Train the trainingPairs onc.
   * Returns true if the data converged in this training iteration, or false otherwise.
   */
  private boolean trainOnce(List<SentencePair> trainingPairs,
                         Set<String> allSourceWords, HashMap coOccurringTargetWords) {
    CounterMap<String, String> sourceTargetCooccurrenceCount = new CounterMap<String, String>();
    Counter<String> totalCooccurrenceCount = new Counter<String>();

    CounterMap<Integer, Integer> alignmentCount = new CounterMap<Integer, Integer>();
    Counter<Integer> totalAlignmentCount = new Counter<Integer>();

    for (SentencePair pair : trainingPairs) {
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();

      int numTargetWords = targetWords.size();
      int numSourceWords = sourceWords.size();
      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        double denom = 0.0;

        for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
          denom += sourceTargetProbability.getCount(sourceWords.get(srcIndex),
                                                    targetWords.get(tgtIndex)) *
                    alignmentProbability.getCount(srcIndex, tgtIndex);
        }

        for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {        
          double posterior_num = sourceTargetProbability.getCount(sourceWords.get(srcIndex),
               targetWords.get(tgtIndex)) * alignmentProbability.getCount(srcIndex, tgtIndex); 
          double posterior = posterior_num / denom;
          sourceTargetCooccurrenceCount.incrementCount(sourceWords.get(srcIndex),
                                                      targetWords.get(tgtIndex),
                                                      posterior);
          totalCooccurrenceCount.incrementCount(targetWords.get(tgtIndex), posterior);
          totalAlignmentCount.incrementCount(srcIndex, posterior);
          alignmentCount.incrementCount(srcIndex, tgtIndex, posterior);
        }
      }
    }

    int numConverged = 0;
    int totalData = 0;
    for (String sourceWord : allSourceWords) {
      for (String targetWord : (HashSet<String>)(coOccurringTargetWords.get(sourceWord))) {

        double newValue = sourceTargetCooccurrenceCount.getCount(
                         sourceWord, targetWord) / totalCooccurrenceCount.getCount(targetWord);

        if (Math.abs(sourceTargetProbability.getCount(sourceWord, targetWord) -
                     newValue) < 0.000001) {
            numConverged++;
    
        }
        totalData++;
        sourceTargetProbability.setCount(sourceWord, targetWord, newValue);
      }
    }

    for (int j = 0; j < maxNumTargetWords; j++) {
      for (int i = 0; i < maxNumSourceWords; i++) {
        double newValue = alignmentCount.getCount(i, j) / totalAlignmentCount.getCount(i);
        alignmentProbability.setCount(i, j, newValue);
      }
    }
    return (numConverged > 0.95 * totalData);
  }

  private void initProbabilities(List<SentencePair> trainingPairs, 
                                 Set<String> allSourceWords,
                                 HashMap coOccurringTargetWords) {
    Model1 model1 = new Model1();
    
    alignmentProbability = new CounterMap<Integer, Integer>();
    sourceTargetProbability = model1.getTranslationProbability(trainingPairs);

    maxNumSourceWords = 0;
    maxNumTargetWords = 0;
    for (SentencePair pair : trainingPairs) {
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();
      maxNumSourceWords = Math.max(maxNumSourceWords, sourceWords.size());
      maxNumTargetWords = Math.max(maxNumTargetWords, targetWords.size());
    }

    for (int j = 0; j < maxNumTargetWords; j++) {
      double denom = 0;
      for (int i = 0; i < maxNumSourceWords; i++) {
        double temp = Math.random();
        denom += temp;
        alignmentProbability.setCount(i, j, temp);
      }

      // Normalize random probability

      for (int i = 0; i < maxNumSourceWords; i++) {
        alignmentProbability.setCount(i, j,
                        alignmentProbability.getCount(i,j) / denom);
      }
    }
  }
}
