
# coding: utf-8

# # HMM词性标注


import nltk
import sys
from nltk.corpus import brown

print(brown.tagged_sents()[0])
print(len(brown.tagged_sents()[0]))

#预处理，添加start和end tag
brown_tags_words = [ ]
for sent in brown.tagged_sents():
    brown_tags_words.append( ("START", "START") )
    brown_tags_words.extend([ (tag[:2], word) for (word, tag) in sent ])
    brown_tags_words.append( ("END", "END") )

#观测概率矩阵B P(wi | ti) = count(wi, ti) / count(ti)  tag为ti，观测到wi的概率
# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
print("The probability of an adjective (VB) being 'beat' is", cpd_tagwords["VB"].prob("beat"))

#状态转移矩阵A P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})  tag为t（i-1）转移到ti的概率
#与words无关，只取出tag
brown_tags = [tag for (tag, word) in brown_tags_words ]
# bigram,考虑前后关系
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))


#已知观测序列“I want to race”及状态序列“PP VB TO VB”，以及模型参数状态转移矩阵cpd_tags,观测概率矩阵cpd_tagwords
#求观测序列与状态序列匹配的概率P(O|I,lambda)
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") *     cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("want") *     cpd_tags["VB"].prob("TO") * cpd_tagwords["TO"].prob("to") *     cpd_tags["TO"].prob("VB") * cpd_tagwords["VB"].prob("race") *     cpd_tags["VB"].prob("END")

print( "The probability of the tag sequence 'START PP VB TO VB END' for 'I want to race' is:", prob_tagsequence)


# # Viterbi实现
# 已知模型参数（初始状态概率未知），已知观测序列，求最好的隐状态

distinct_tags = set(brown_tags)

sentence = ["I", "want", "to", "race"]
len_sent = len(sentence)

#初始化Pi 初始状态概率向量
viterbi = []
back_pointer = []

first_viterbi = {} #记录以各个tag开始，观测到sentence[0]的概率
first_backpointer = {}
for tag in distinct_tags:
    if tag=="START":
        continue
    first_viterbi[tag] = cpd_tags["START"].prob(tag)*cpd_tagwords[tag].prob(sentence[0])
    first_backpointer[tag] = "START" #记录以当前tag结束且概率最大的路径的前一个tag
viterbi.append(first_viterbi)
back_pointer.append(first_backpointer)


for wordindex in range(1,len_sent):
    this_vertebi = {} #记录观测序列到sentence[wordindex]时，以各个tag结束的最大概率
    this_backpointer = {} #记录观测序列到sentence[wordindex]时，以各个tag结束的最大概率，tag的前一个tag
    previous_viterbi = viterbi[-1]
    for tag in distinct_tags:
        if tag=="START":
            continue
        best_previous = max(previous_viterbi.keys(),key=lambda pretag:previous_viterbi[pretag]* \
                            cpd_tags[pretag].prob(tag)*cpd_tagwords[tag].prob(sentence[wordindex]))
        this_vertebi[ tag ] = previous_viterbi[best_previous]*cpd_tags[best_previous].prob(tag) *cpd_tagwords[tag].prob(sentence[wordindex])
        this_backpointer[tag] = best_previous
    viterbi.append(this_vertebi)
    back_pointer.append(this_backpointer)

#找到"END"同上处理
previous_viterbi = viterbi[-1]
best_previous = max(previous_viterbi.keys(),key=lambda pretag:previous_viterbi[pretag]* \
                            cpd_tags[pretag].prob("END"))

maximum_prob = previous_viterbi[best_previous]*cpd_tags[best_previous].prob("END")


best_tagsequence = ["END",best_previous]
back_pointer.reverse()

#回溯
current_best_tag = best_previous
for bp in back_pointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]
    if current_best_tag=="START":
        break
best_tagsequence.reverse()
print("best_tagsequence",best_tagsequence)
print("maximum_prob",maximum_prob)
"""
output:
[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL'), ('said', 'VBD'), ('Friday', 'NR'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'IN'), ("Atlanta's", 'NP$'), ('recent', 'JJ'), ('primary', 'NN'), ('election', 'NN'), ('produced', 'VBD'), ('``', '``'), ('no', 'AT'), ('evidence', 'NN'), ("''", "''"), ('that', 'CS'), ('any', 'DTI'), ('irregularities', 'NNS'), ('took', 'VBD'), ('place', 'NN'), ('.', '.')]
25
The probability of an adjective (VB) being 'beat' is 0.0003194005628355864
If we have just seen 'VB', the probability of 'NN' is 0.10970977711020183
The probability of the tag sequence 'START PP VB TO VB END' for 'I want to race' is: 1.0817766461150474e-14
best_tagsequence ['START', 'PP', 'VB', 'IN', 'NN', 'END']
maximum_prob 5.71772824864617e-14
"""

































